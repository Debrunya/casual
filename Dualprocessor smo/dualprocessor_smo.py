# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:42:28 2022

@author: znaha
"""
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from prettytable import PrettyTable

def _cls() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')



def solver_1(conditions, server):
    """
    выбирает требование из очереди с максимальным числом требований,
        при этом, если есть очереди с одинаковым числом требований, выбирает очередь с наименьшим номером
    """
    
    index = np.argmax([conditions[2], conditions[3], conditions[4]])
    conditions[server] = f'g{index + 1}'
    conditions[index + 2] -= 1
    return conditions


def solver_2(conditions, server):
    """
    выбирает требование из очередей с приоритетами:
        из первой очереди - первый приоритет
        из второй очереди - второй приоритет
        из третий очереди - третий приоритет
    """
    
    if conditions[2] != 0:
        conditions[server] = 'g1'
        conditions[2] -= 1
        
    elif conditions[3] != 0:
        conditions[server] = 'g2'
        conditions[3] -= 1
    
    else:
        conditions[server] = 'g3'
        conditions[4] -= 1
    return conditions


def solver_3(conditions, server):
    return conditions


def second_stage(conditions, queue, probs, count_apt, count_whole):
    """
    определим, что происходит с требованием после обслуживания на приборе:
        покинуло ли требование систему или перешло в конкретную очередь на повторное обслуживание
    """
    
    if queue == 'g1':
        index = np.random.choice([2, 3, 4, -1], p=[probs[0], probs[1], probs[2], 1-(probs[0]+probs[1]+probs[2])])
        if index != -1:
            conditions[index] += 1
            count_apt[index-2] += 1
        count_whole[0] += 1
            
    elif queue == 'g2':
        index = np.random.choice([2, 3, 4, -1], p=[probs[3], probs[4], probs[5], 1-(probs[3]+probs[4]+probs[5])])
        if index != -1:
            conditions[index] += 1
            count_apt[index+1] += 1
        count_whole[1] += 1
            
    else:
        index = np.random.choice([2, 3, 4, -1], p=[probs[6], probs[7], probs[8], 1-(probs[6]+probs[7]+probs[8])])
        if index != -1:
            conditions[index] += 1
            count_apt[index+4] += 1
        count_whole[2] += 1
    return conditions, count_apt, count_whole


def main():
    start_time = time.time()
    np.random.seed()
    
    """
    массив показателей экспоненциальных распределений liambdas = [-1.0, Переналадка, П1, П2, П3, Б1, Б2, Б3]
    
    массив вероятностного распределения Pjr (вероятность с которой требование после обслуживания
        переходит в любую из трех очередей) probabilities = [p11, p12, p13, p21, p22, p23, p31, p31, p33]
    
    массив переменных таймеров до изменения состояния times = [ПР1, ПР2, П1, П2, П3]
    
    модельное время mod_time, максимальное время наблюдения N
    
    массив состояний приборов и очередей conditions = ['Г0', 'Г0', O1, O2, O3]
                                         Г0 - простой
                                         Г1 - обслуживание требования первого потока
                                         Г2 - обслуживание требования второого потока
                                         Г3 - обслуживание требования третьего потока
                                         Г4 - переналадка
    
    массив для сбора статистических данных stats = [t1w, t2w]
                                           t1w - время работы первого прибора
                                           t2w - время работы второго прибора
    
    массивы для сбора статистических данных, нужных для проверки входных параметров распределений
        (лямбд экспоненциальных распределений)
        сумма времен time_stats = [-1.0, Переналадки, П1, П2, П3, Б1, Б2, Б3]
        количество сгенерированных сл. в. count_stats = [-1.0, Переналадки, П1, П2, П3, Б1, Б2, Б3]
    
    массивы для сбора статистических данных, нужных для проверки входных параметров вероятностного распределения Pjr
        количество исходов, когда требования возвращалось в конкретную очередь count_apt = [p11, p12, p13, p21, p22, p23, p31, p31, p33]
        количество всех исходов count_whole = [p1, p2, p3]
    """
    
    liambdas = np.array([-1.0, 5.0, 1.22, 0.8, 0.75, 4.85, 2.15, 7.95])
    probabilities = np.array([0.02, 0.07, 0.05, 0.05, 0.02, 0.07, 0.07, 0.05, 0.02])
    N = 10**6
    
    times = np.array([10.0**9, 10.0**9, 0.0, 0.0, 0.0])
    conditions = ['g0', 'g0', 0, 0, 0]
    
    mod_time = 0.0
    stats = np.array([0.0, 0.0])
    time_stats = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    count_stats = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    count_apt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    count_whole = np.array([0, 0, 0])
    
    x = st.expon.rvs(scale = 1/liambdas[2]) #время до прихода требования по 1 потоку
    times[2] = x
    time_stats[2] += x
    count_stats[2] += 1
    x = st.expon.rvs(scale = 1/liambdas[3]) #время до прихода требования по 2 потоку
    times[3] = x
    time_stats[3] += x
    count_stats[3] += 1
    x = st.expon.rvs(scale = 1/liambdas[4]) #время до прихода требования по 3 потоку
    times[4] = x
    time_stats[4] += x
    count_stats[4] += 1

    """
    print(times)
    print(conditions)
    print('\n')
    """

    while mod_time <= N:
        """
        поиск ближайшего по времени изменения состояния системы, его обработка
        """
        index = np.argmin(times)
        
        if index >= 2: #пришло требование по потоку
            mod_time += times[index]
            times -= times[index] #добавим пройденное время в модальное и вычтем его из всех таймеров
            x = st.expon.rvs(scale = 1/liambdas[index])
            times[index] = x #обновмим время до прихода следующего требования по потоку
            time_stats[index] += x
            count_stats[index] += 1
            conditions[index] += 1 #добавим пришедшее требование в очередь
        
        else: #изменение состояния прибора
            #если закончилось обслуживание, то уходим в переналадку
            if conditions[index] == 'g1' or conditions[index] == 'g2' or conditions[index] == 'g3':
                mod_time += times[index]
                times -= times[index]
                x = st.expon.rvs(scale = 1/liambdas[1]) #определим время переналадки
                times[index] = x
                stats[index] += x
                time_stats[1] += x
                count_stats[1] += 1
                conditions, count_apt, count_whole = second_stage(conditions, conditions[index], probabilities, count_apt, count_whole)
                conditions[index] = 'g4'
            
            elif conditions[index] == 'g4': #если закончилась переналадка, то поиск нового требования на обслуживание
                mod_time += times[index]
                times -= times[index]
                if conditions[2] != 0 or conditions[3] != 0 or conditions[4] != 0: #если есть требования в очередях, вызовем h(x)
                    conditions = solver_1(conditions, index) #первый вариант h(x)
                    #conditions = solver_2(conditions, index) #второй вариант h(x)
                    if conditions[index] == 'g1':
                        x = st.expon.rvs(scale = 1/liambdas[5])
                        times[index] = x
                        stats[index] += x
                        time_stats[5] += x
                        count_stats[5] += 1
                    
                    elif conditions[index] == 'g2':
                        x = st.expon.rvs(scale = 1/liambdas[6])
                        times[index] = x
                        stats[index] += x
                        time_stats[6] += x
                        count_stats[6] += 1
                    
                    elif conditions[index] == 'g3':
                        x = st.expon.rvs(scale = 1/liambdas[7])
                        times[index] = x
                        stats[index] += x
                        time_stats[7] += x
                        count_stats[7] += 1
                        
                else:  #если очереди пусты, определим состояние простоя
                    conditions[index] = 'g0'
                    times[index] = 10.0**9
        
        """
        вывод приборов из простоя
        """
        if conditions[0] == 'g0':
            if conditions[2] != 0 or conditions[3] != 0 or conditions[4] != 0: #если есть требования в очередях, вызовем h(x)
                conditions = solver_1(conditions, 0) #первый вариант h(x)
                #conditions = solver_2(conditions, 0) #второй вариант h(x)
                if conditions[0] == 'g1':
                    x = st.expon.rvs(scale = 1/liambdas[5])
                    times[0] = x
                    stats[0] += x
                    time_stats[5] += x
                    count_stats[5] += 1
                
                elif conditions[0] == 'g2':
                    x = st.expon.rvs(scale = 1/liambdas[6])
                    times[0] = x
                    stats[0] += x
                    time_stats[6] += x
                    count_stats[6] += 1
                
                elif conditions[0] == 'g3':
                    x = st.expon.rvs(scale = 1/liambdas[7])
                    times[0] = x
                    stats[0] += x
                    time_stats[7] += x
                    count_stats[7] += 1
        
        if conditions[1] == 'g0':
            if conditions[2] != 0 or conditions[3] != 0 or conditions[4] != 0: #если есть требования в очередях, вызовем h(x)
                conditions = solver_1(conditions, 1) #первый вариант h(x)
                #conditions = solver_2(conditions, 1) #второй вариант h(x)
                if conditions[1] == 'g1':
                    x = st.expon.rvs(scale = 1/liambdas[5])
                    times[1] = x
                    stats[1] += x
                    time_stats[5] += x
                    count_stats[5] += 1
                
                elif conditions[1] == 'g2':
                    x = st.expon.rvs(scale = 1/liambdas[6])
                    times[1] = x
                    stats[1] += x
                    time_stats[6] += x
                    count_stats[6] += 1
                
                elif conditions[1] == 'g3':
                    x = st.expon.rvs(scale = 1/liambdas[7])
                    times[1] = x
                    stats[1] += x
                    time_stats[7] += x
                    count_stats[7] += 1
       
        
       
        """
        print(times)
        print(conditions, mod_time)
        print(stats, stats/mod_time)
        print('\n')
        """
    
    
    
    #вывод статистических данных
    table1 = PrettyTable()
    table1.field_names = ['Модельное вр.', 'Активн. вр. ПР1', 'Активн. вр. ПР2', 'О1', 'О2', 'О3']
    table1.add_row([f'{mod_time:.10f}', f'{stats[0]/mod_time:.10f}', f'{stats[1]/mod_time:.10f}', f'{conditions[2]}',\
                    f'{conditions[3]}', f'{conditions[4]}'])
    print(table1, '\n')
    
    table2 = PrettyTable()
    table2.field_names = ['lambda', 'Переналадки', 'П1', 'П2', 'П3', 'Б1', 'Б2', 'Б3']
    table2.add_row(['Значение', f'{count_stats[1]/time_stats[1]:.5f}', f'{count_stats[2]/time_stats[2]:.5f}',\
                    f'{count_stats[3]/time_stats[3]:.5f}', f'{count_stats[4]/time_stats[4]:.5f}',\
                    f'{count_stats[5]/time_stats[5]:.5f}', f'{count_stats[6]/time_stats[6]:.5f}', f'{count_stats[7]/time_stats[7]:.5f}'])
    print(table2, '\n')
    
    table3 = PrettyTable()
    table3.field_names = ['p11', 'p12', 'p13', 'p21', 'p22', 'p23', 'p31', 'p32', 'p33']
    table3.add_row([f'{count_apt[0]/count_whole[0]:.5f}', f'{count_apt[1]/count_whole[0]:.5f}', f'{count_apt[2]/count_whole[0]:.5f}',\
                    f'{count_apt[3]/count_whole[1]:.5f}', f'{count_apt[4]/count_whole[1]:.5f}', f'{count_apt[5]/count_whole[1]:.5f}',\
                    f'{count_apt[6]/count_whole[2]:.5f}', f'{count_apt[7]/count_whole[2]:.5f}', f'{count_apt[8]/count_whole[2]:.5f}'])
    print(table3, '\n')
    
    
    hours = int((time.time()-start_time)//3600)
    minutes = int((time.time()-start_time-3600*hours)//60)
    seconds = time.time()-start_time-3600*hours-60*minutes
    print(f'Время выполнения {hours} hours {minutes} minutes {seconds} seconds')


if __name__ == "__main__":
    main()
    