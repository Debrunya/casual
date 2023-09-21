# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:33:02 2023

@author: znaha
"""


import time as tm #засечь время исполнения программы
import numpy as np #работа с массивами на C
import scipy.stats as st #библиотека для научных вычислений(генераторы случайных величин)

from prettytable import PrettyTable #нужна будет для табличного вывода статистики


def pack_length_generator(R, G):
    """
    Генератор длины пачки.
    На входе параметры распределения Бартлетта, на выходе длина пачки.
    Работает по принципу метода для получения Пуассона.
    """
    #введем значения параметров, которые будут использоваться в реккурентных формулах при k>2
    pack_len = 2
    prob_summ = (1-R)+R*(1-G) #сумма вероятностей поступления любого количества от 0 до k машин в пачке
    
    #проверки для k=1, k=2, k>2; когда попадаем в нужный отрезок внутри [0,1] - останавливаемся 
    p = np.random.uniform(0, 1)
    if p < 1-R:  #проверка для k=1
        pack_len = 1 
    elif p < (1-R)+R*(1-G):  #проверка для k=2
        pack_len = 2
    else:
        while p >= prob_summ: #итеративный процесс для k>2, выходит из цикла, как превысит порог вероятности(см. Ф.Р.)
            pack_len += 1
            prob_summ += R*(1-G)*G**(pack_len-2)
    
    return pack_len


def bartlett_stream(liambda, R, G, N):
    """
    Генератор потока Бартлетта для задаваемых параметров распределения и длительности моделирования.
    На входе общая интенсивность по потоку, параметры распределения Бартлетта, 
    длительность моделирования, на выходе времена поступления машин внутри времени моделирования.
    """
    time = np.array([])
    
    liambda_packs = liambda*(1-G)/(1-G+R) #лямбда для медленных машин
    slow_cars = st.poisson.rvs(liambda_packs*N)
    if slow_cars < 1: #когда за время N приходит ноль машин: отправляем маркер, что пришло ноль машин за период
        time = np.append(time, None)
        return time
    else: #иначе генирируем времена поступления машин: сначала медленных, потом по ним быстрых
        time = np.random.uniform(0, N, slow_cars)
        time = np.sort(time) #время медленных
    
        fast_cars = pack_length_generator(R, G)-1
        fast_cars_counter = np.array([fast_cars])
        delta_min = N-time[slow_cars-1]
        if slow_cars > 1: #для каждой медленной получим количество быстрых в пачке и просуммируем
            for i in range(slow_cars-1):
                tmp = time[i+1]-time[i]
                if delta_min > tmp: delta_min = tmp
                plgen_for_one_slow_car = pack_length_generator(R, G)-1
                fast_cars_counter = np.append(fast_cars_counter, plgen_for_one_slow_car)
                fast_cars += plgen_for_one_slow_car
        
        delta = delta_min/(fast_cars+2) #время между быстрыми машинами
        for i in range(slow_cars-1, -1, -1):
            j = 0
            k = i
            while j < fast_cars_counter[i]:
                time = np.insert(time, k+1, time[k]+delta) #расставляем быстрые машины
                k += 1
                j += 1
                
        return time[::-1]


def experiment(T1, T3, T_mod):
    np.random.seed()
    
    """
    Блок ввода параметров системы:
        liambdas - [П1, П2] - интенсивности потоков П1 и П2
        bartlett_params - [R1, G1, R2, G2] - параметры R и G распределения Бартлетта для потоков П1 и П2
        condition_times - [Г1, Г2, Г3, Г4, Г5, Г6] - фиксированные времена пребывания в состояниях Г1-Г6
        streams_bandwidth - [Г1, Г3, Г5, Г6] - пропускная способность потоков насыщения в режимах зеленого света по потокам (машин/такт)
        bandwith_time - [Г1, Г3, Г5, Г6] - пропускная способность потоков насыщения в режимах зеленого света по потокам (секунд/машину)
        max_stream_queue - [П1, П2] - максимальная возможная очередь по одному потоку, при котором может включится дообслуживание
                                        по другому потоку
        T_mod - время, в течение которого будет происходить моделирование
        model_time - модельное время
        start_condition - [Гn, [x1, x2]] - начальное состояние системы: режим О.У.(1-6), очереди по потокам П1 и П2
        condition - [Гn, [x1, x2]]
        
    Статистика:
        stats - [G1, кол-во 1, G2, кол-во 2] - суммарное время ожидания по потокам П1 и П2 и количество требований, которые ожидали
        waiting_queue(1, 2) - массивы времен поступления требований в режимах ожидания по потокам П1 и П2
    """
    liambdas = np.array([0.3, 0.2])
    bartlett_params = np.array([0.5, 0.5, 0.5, 0.5])
    condition_times = np.array([T1, 5, T3, 5, 10, 10])
    bandwith_time = np.array([1, 1, 2, 2])
    streams_bandwidth = np.array([int(T1/bandwith_time[0]), int(T3/bandwith_time[1]), int(condition_times[4]/bandwith_time[2]),
                                  int(condition_times[5]/bandwith_time[3])])
    max_stream_queue = np.array([10, 10])
    model_time = 0
    start_condition = np.array([1, np.array([0, 0])], dtype=object)
    condition = start_condition
    
    stats = np.array([0., 0, 0., 0])
    waiting_queue_1 = np.array([])
    waiting_queue_2 = np.array([])
    
    #запустим итеративный процесс слежки за системой в моменты смены режимов О.У.
    while model_time < T_mod:
        """
        1) Определяем текущее состояние системы и время до следующего скачка.
        2) Моделируем потоки за время в этом текущем состоянии.
        3) Если состояние позволяет обслужить потоки, то обслуживаем.
        4) Принимаем решение о следующем режиме О.У.
        5) Записываем состояние системы в массив и повторяем цикл.
        """
        jump_time = condition_times[condition[0]-1] #время нахождения в текущем состоянии О.У. до смены режима 
        
        if condition[1][0] > 200 or condition[1][1] > 200: #если нет стационара, то выходим из цикла
            condition[0] = -1
            break
        
        if condition[0] == 1: #зеленый по первому потоку в основной фазе(не дообслуживание)
            first_stream_cars_time = bartlett_stream(liambdas[0], bartlett_params[0], bartlett_params[1], jump_time + condition_times[1])
            second_stream_cars_time = bartlett_stream(liambdas[1], bartlett_params[2], bartlett_params[3], jump_time + condition_times[1])
            first_stream_cars = 0
            second_stream_cars = 0
            serviced_cars = 0
            w_q_1_size = waiting_queue_1.size
            
            #определим количество пришедших машин за скачек по потокам
            if first_stream_cars_time[0] == None:
                first_stream_cars_time = np.array([])
            else:
                first_stream_cars = first_stream_cars_time.size
            if second_stream_cars_time[0] == None:
                second_stream_cars_time = np.array([])
            else:
                second_stream_cars = second_stream_cars_time.size
            
            #определим очереди по потокам после режима Г1
            tmp = condition[1][0]
            condition[1][0] = max(tmp + first_stream_cars - streams_bandwidth[0], 0)
            condition[1][1] = condition[1][1] + second_stream_cars
            if condition[1][0] == 0:
                serviced_cars = tmp + first_stream_cars
            else:
                serviced_cars = streams_bandwidth[0]
            
            #запишем статистические величины
            if w_q_1_size > 0:
                for i in range(w_q_1_size):
                    waiting_queue_1[i] += condition_times[3] #прибавление к времени ожидания фазу желтого
                
                if w_q_1_size > serviced_cars:
                    stats[1] += serviced_cars
                    stats[0] += (serviced_cars-1)*bandwith_time[0]*serviced_cars/2 #время, которые ждут заявки в фазе обслуживания
                    tmp_slice = waiting_queue_1[:serviced_cars]
                    stats[0] += np.sum(tmp_slice)
                    tmp = np.arange(serviced_cars)
                    waiting_queue_1 = np.delete(waiting_queue_1, tmp)
                    waiting_queue_1 += jump_time
                    waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
                
                else:
                    stats[1] += serviced_cars
                    stats[0] += (w_q_1_size-1)*bandwith_time[0]*w_q_1_size/2
                    stats[0] += np.sum(waiting_queue_1)
                    waiting_queue_1 = np.array([])
                    
                    if first_stream_cars > serviced_cars-w_q_1_size:
                        tmp = np.arange(serviced_cars-w_q_1_size)
                        first_stream_cars_time = np.delete(first_stream_cars_time, tmp)
                        waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
                        
            elif first_stream_cars > serviced_cars:
                stats[1] += serviced_cars
                tmp = np.arange(serviced_cars)
                first_stream_cars_time = np.delete(first_stream_cars_time, tmp)
                waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
            
            else:
                stats[1] += serviced_cars
                        
            waiting_queue_2 += (jump_time + condition_times[3]) #обработаем второй поток
            waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
            
            #определим следующее состояние
            if condition[1][0] > 0 and condition[1][1] < max_stream_queue[1]: #дообслужим первый поток, если позволяет очередь по второму потоку
                condition[0] = 5
            else: #дообслуживание невозможно, включим желтый
                condition[0] = 2
            
            
        
        elif condition[0] == 3: #зеленый по второму потоку в основной фазе(не дообслуживание)
            first_stream_cars_time = bartlett_stream(liambdas[0], bartlett_params[0], bartlett_params[1], jump_time + condition_times[3])
            second_stream_cars_time = bartlett_stream(liambdas[1], bartlett_params[2], bartlett_params[3], jump_time + condition_times[3])
            first_stream_cars = 0
            second_stream_cars = 0
            serviced_cars = 0
            w_q_2_size = waiting_queue_2.size
            
            #определим количество пришедших машин за скачек по потокам
            if first_stream_cars_time[0] == None:
                first_stream_cars_time = np.array([])
            else:
                first_stream_cars = first_stream_cars_time.size
            if second_stream_cars_time[0] == None:
                second_stream_cars_time = np.array([])
            else:
                second_stream_cars = second_stream_cars_time.size
                
            #определим очереди по потокам после режима Г3
            tmp = condition[1][1]
            condition[1][0] = condition[1][0] + first_stream_cars
            condition[1][1] = max(tmp + second_stream_cars - streams_bandwidth[1], 0)
            if condition[1][1] == 0:
                serviced_cars = tmp + second_stream_cars
            else:
                serviced_cars = streams_bandwidth[1]

            #запишем статистические величины
            if w_q_2_size > 0:
                for i in range(w_q_2_size):
                    waiting_queue_2[i] += condition_times[1] #прибавление к времени ожидания фазу желтого
                
                if w_q_2_size > serviced_cars:
                    stats[3] += serviced_cars
                    stats[2] += (serviced_cars-1)*bandwith_time[1]*serviced_cars/2 #время, которые ждут заявки в фазе обслуживания
                    tmp_slice = waiting_queue_2[:serviced_cars]
                    stats[2] += np.sum(tmp_slice)
                    tmp = np.arange(serviced_cars)
                    waiting_queue_2 = np.delete(waiting_queue_2, tmp)
                    waiting_queue_2 += jump_time
                    waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
                
                else:
                    stats[3] += serviced_cars
                    stats[2] += (w_q_2_size-1)*bandwith_time[1]*w_q_2_size/2
                    stats[2] += np.sum(waiting_queue_2)
                    waiting_queue_2 = np.array([])
                    
                    if second_stream_cars > serviced_cars-w_q_2_size:
                        tmp = np.arange(serviced_cars-w_q_2_size)
                        second_stream_cars_time = np.delete(second_stream_cars_time, tmp)
                        waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
                        
            elif second_stream_cars > serviced_cars:
                stats[3] += serviced_cars
                tmp = np.arange(serviced_cars)
                second_stream_cars_time = np.delete(second_stream_cars_time, tmp)
                waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
            
            else:
                stats[3] += serviced_cars
                        
            waiting_queue_1 += (jump_time + condition_times[1]) #обработаем первый поток
            waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
                           
            #определим следующее состояние
            if condition[1][1] > 0 and condition[1][0] < max_stream_queue[0]: #дообслужим второй поток, если позволяет очередь по первому потоку
                condition[0] = 6
            else: #дообслуживание невозможно, включим желтый
                condition[0] = 4
                
        
        elif condition[0] == 5: #зеленый по первому потоку в дообслуживающей фазе
            first_stream_cars_time = bartlett_stream(liambdas[0], bartlett_params[0], bartlett_params[1], jump_time)
            second_stream_cars_time = bartlett_stream(liambdas[1], bartlett_params[2], bartlett_params[3], jump_time)
            first_stream_cars = 0
            second_stream_cars = 0
            serviced_cars = 0
            w_q_1_size = waiting_queue_1.size
            
            #определим количество пришедших машин за скачек по потокам
            if first_stream_cars_time[0] == None:
                first_stream_cars_time = np.array([])
            else:
                first_stream_cars = first_stream_cars_time.size
            if second_stream_cars_time[0] == None:
                second_stream_cars_time = np.array([])
            else:
                second_stream_cars = second_stream_cars_time.size
            
            #определим очереди по потокам после режима Г1
            tmp = condition[1][0]
            condition[1][0] = max(tmp + first_stream_cars - streams_bandwidth[2], 0)
            condition[1][1] = condition[1][1] + second_stream_cars
            if condition[1][0] == 0:
                serviced_cars = tmp + first_stream_cars
            else:
                serviced_cars = streams_bandwidth[2]
            
            if w_q_1_size > 0:
                if w_q_1_size > serviced_cars:
                    stats[1] += serviced_cars
                    stats[0] += (serviced_cars-1)*bandwith_time[2]*serviced_cars/2 #время, которые ждут заявки в фазе обслуживания
                    tmp_slice = waiting_queue_1[:serviced_cars]
                    stats[0] += np.sum(tmp_slice)
                    tmp = np.arange(serviced_cars)
                    waiting_queue_1 = np.delete(waiting_queue_1, tmp)
                    waiting_queue_1 += jump_time
                    waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
                
                else:
                    stats[1] += serviced_cars
                    stats[0] += (w_q_1_size-1)*bandwith_time[2]*w_q_1_size/2
                    stats[0] += np.sum(waiting_queue_1)
                    waiting_queue_1 = np.array([])
                    
                    if first_stream_cars > serviced_cars-w_q_1_size:
                        tmp = np.arange(serviced_cars-w_q_1_size)
                        first_stream_cars_time = np.delete(first_stream_cars_time, tmp)
                        waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
                        
            elif first_stream_cars > serviced_cars:
                stats[1] += serviced_cars
                tmp = np.arange(serviced_cars)
                first_stream_cars_time = np.delete(first_stream_cars_time, tmp)
                waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
            
            else:
                stats[1] += serviced_cars
                        
            waiting_queue_2 += jump_time #обработаем второй поток
            waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
            
            #следующее состояние "желтый свет"
            condition[0] = 2
                
        
        elif condition[0] == 6: #зеленый по второму потоку в дообслуживающей фазе
            first_stream_cars_time = bartlett_stream(liambdas[0], bartlett_params[0], bartlett_params[1], jump_time)
            second_stream_cars_time = bartlett_stream(liambdas[1], bartlett_params[2], bartlett_params[3], jump_time)
            first_stream_cars = 0
            second_stream_cars = 0
            serviced_cars = 0
            w_q_2_size = waiting_queue_2.size
            
            #определим количество пришедших машин за скачек по потокам
            if first_stream_cars_time[0] == None:
                first_stream_cars_time = np.array([])
            else:
                first_stream_cars = first_stream_cars_time.size
            if second_stream_cars_time[0] == None:
                second_stream_cars_time = np.array([])
            else:
                second_stream_cars = second_stream_cars_time.size
                
            #определим очереди по потокам после режима Г3
            tmp = condition[1][1]
            condition[1][0] = condition[1][0] + first_stream_cars
            condition[1][1] = max(tmp + second_stream_cars - streams_bandwidth[3], 0)
            if condition[1][1] == 0:
                serviced_cars = tmp + second_stream_cars
            else:
                serviced_cars = streams_bandwidth[3]

            #запишем статистические величины
            if w_q_2_size > 0:
                if w_q_2_size > serviced_cars:
                    stats[3] += serviced_cars
                    stats[2] += (serviced_cars-1)*bandwith_time[3]*serviced_cars/2 #время, которые ждут заявки в фазе обслуживания
                    tmp_slice = waiting_queue_2[:serviced_cars]
                    stats[2] += np.sum(tmp_slice)
                    tmp = np.arange(serviced_cars)
                    waiting_queue_2 = np.delete(waiting_queue_2, tmp)
                    waiting_queue_2 += jump_time
                    waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
                
                else:
                    stats[3] += serviced_cars
                    stats[2] += (w_q_2_size-1)*bandwith_time[3]*w_q_2_size/2
                    stats[2] += np.sum(waiting_queue_2)
                    waiting_queue_2 = np.array([])
                    
                    if second_stream_cars > serviced_cars-w_q_2_size:
                        tmp = np.arange(serviced_cars-w_q_2_size)
                        second_stream_cars_time = np.delete(second_stream_cars_time, tmp)
                        waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
                        
            elif second_stream_cars > serviced_cars:
                stats[3] += serviced_cars
                tmp = np.arange(serviced_cars)
                second_stream_cars_time = np.delete(second_stream_cars_time, tmp)
                waiting_queue_2 = np.append(waiting_queue_2, second_stream_cars_time)
            
            else:
                stats[3] += serviced_cars
                        
            waiting_queue_1 += jump_time #обработаем первый поток
            waiting_queue_1 = np.append(waiting_queue_1, first_stream_cars_time)
            
            #следующее состояние "желтый свет"
            condition[0] = 4
                
        
        else: #желтый свет
            #определим следующее состояние
            if condition[0] == 2: #если у нас желтый после зеленого по первому потоку, переключимся в Г3
                condition[0] = 3
            else: #если у нас желтый после зеленого по второму потоку, переключимся в Г1
                condition[0] = 1
                
                
        #запишем время скачка в модельное время и зациклим
        model_time += jump_time
        
        #print(condition) #чтобы смотреть динамику системы
    
    return stats, liambdas, condition
    
    
def main():
    start_time = tm.time() #время старта рассчетов, чтобы потом посчитать длительность моделирования в реальном времени
    
    """
    stats, liambdas, cond = experiment(40, 20, 10**6) #ввод Т1 и Т3 в аргументы
    
    #статистика
    gamma_1 = stats[0]/stats[1]
    gamma_2 = stats[2]/stats[3]
    gamma = (liambdas[0]*stats[0]/stats[1]+liambdas[1]*stats[2]/stats[3])/(np.sum(liambdas))
    
    print(f'gamma_1 = {gamma_1}, gamma_2 = {gamma_2}')
    print(f'gamma = {gamma}')
    """
    
    #поиск квазиоптимального режима методом перебора
    """
    Нужно сходить в функцию experiment(строчка 100), задать параметры системы
    Задать параметры сетки перебора start_T, end_T, step
    В представлении матрицы:
        ось x - ось времени T1 идет влево
        ось y - ось времени T3 идет вниз
    Остановка уточнения гаммы происходит тогда, когда разница на двух промежутках моделирования становится меньше 1 секунды
            (первый промежуток длины N, второй промежуток длины 2*N)
    """
    start_T = 20
    end_T = 100
    step = 20
    matrix = np.zeros(int(((end_T-start_T)/step+1)**2)).reshape(int((end_T-start_T)/step+1), int((end_T-start_T)/step+1))
    matrix_min = 1000.0
    t1_min  = 0.0
    t3_min = 0.0
    
    m = 1
    for j in range(start_T, end_T, step):
        k = 1
        for i in range(start_T, end_T, step):
            if i+j > 100: #отчерчивание линией t1+t2+t3+t4<150
                matrix[m][k] = -10
                print(matrix)
                continue
            
            N = 10**4
            stats_prev, liambdas, cond_prew = experiment(i, j, N)
            N *= 2
            stats_new, liambdas, cond_new = experiment(i, j, N)
            
            if cond_prew[0] != -1:
                gamma_prev = (liambdas[0]*stats_prev[0]/stats_prev[1]+liambdas[1]*stats_prev[2]/stats_prev[3])/(np.sum(liambdas))
            else:
                matrix[m][k] = -1
                print(matrix)
                k += 1
                continue
    
            if cond_new[0] != -1:
                gamma_new = (liambdas[0]*stats_new[0]/stats_new[1]+liambdas[1]*stats_new[2]/stats_new[3])/(np.sum(liambdas))
            else:
                matrix[m][k] = -1
                print(matrix)
                k += 1
                continue
            
            marker = True
            while abs(gamma_prev-gamma_new) > 1.: #разница между реализациями должна быть меньше заданного числа (выход на точность)
                N *= 2
                stats_prev[0] += stats_new[0]
                stats_prev[1] += stats_new[1]
                stats_prev[2] += stats_new[2]
                stats_prev[3] += stats_new[3]
                gamma_prev = (liambdas[0]*stats_prev[0]/stats_prev[1]+liambdas[1]*stats_prev[2]/stats_prev[3])/(np.sum(liambdas))
                
                stats_new, liambdas, cond_new = experiment(i, j, N)        
                if cond_new[0] != -1:
                    gamma_new = (liambdas[0]*stats_new[0]/stats_new[1]+liambdas[1]*stats_new[2]/stats_new[3])/(np.sum(liambdas))
                else:
                    marker = False
                    break
            
            if marker == True:
                matrix[m][k] = gamma_new
                if gamma_new < matrix_min:
                    matrix_min = gamma_new
                    t1_min = i
                    t3_min = j
            else:
                matrix[m][k] = -1
            print(matrix)
            k += 1
            
        m += 1
    
    k = 1
    for i in range(start_T, end_T, step):
        matrix[0][k] = i
        matrix[k][0] = i
        k += 1  
        
    print(matrix)
    print(f'Минимум матрицы = {matrix_min:.3f}, при T1 = {t1_min} и T3 = {t3_min}')
    
    #вычислим реальное время работы программы
    hours = int((tm.time()-start_time)//3600)
    minutes = int((tm.time()-start_time-3600*hours)//60)
    seconds = tm.time()-start_time-3600*hours-60*minutes
    print(f'Время выполнения {hours} hours {minutes} minutes {seconds:.4f} seconds')
    
    return matrix


if __name__ == "__main__":
    matrix = main()
    
    
    
    
    
    
    