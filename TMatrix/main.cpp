#include <iostream>
using namespace std;

class TMatrix
{
    int n;
    int len;
    int* matrix;

public:
    TMatrix(int _n);
    ~TMatrix();

    int GetN() const;
    int GetLen() const;
    void SetInt(int i, int j, int number);
    int GetInt(int i, int j) const;

    TMatrix operator=(const TMatrix& mat);
    TMatrix operator+ (const TMatrix& mat);

    friend istream& operator>>(istream& istr, TMatrix& mat);
    friend ostream& operator<<(ostream& ostr, const TMatrix& mat);
};

TMatrix::TMatrix(int _n)
{
    n = _n;
    len = n * (n + 1) / 2;
    matrix = new int[len];
    for (int i = 0; i < len; i++)
    {
        matrix[i] = 0;
    }
}

TMatrix::~TMatrix()
{
    delete[] matrix;
}

int TMatrix::GetN() const
{
    return n;
}

int TMatrix::GetLen() const
{
    return len;
}

void TMatrix::SetInt(int _i, int _j, int number)
{
    int i = _i - 1;
    int j = _j - 1;
    int place = i * n - i * (i - 1) / 2 + j - i;
    matrix[place] = number;
    return;
}

int TMatrix::GetInt(int i, int j) const
{
    int place = i * n - i * (i - 1) / 2 + j - i;
    return matrix[place];
}

TMatrix TMatrix::operator=(const TMatrix& mat)
{
    for (int i = 0; i < len; i++)
    {
        if (matrix[i] != mat.matrix[i]) matrix[i] = mat.matrix[i];
    }
    return *this;
}

TMatrix TMatrix::operator+(const TMatrix& mat)
{
    TMatrix temp(n);
    for (int i = 0; i < len; i++)
    {
        temp.matrix[i] = matrix[i] + mat.matrix[i];
    }
    return temp;
}

istream& operator>>(istream& istr, TMatrix& mat)
{
    int _i, _j, _number;

    istr >> _i;
    istr >> _j;
    istr >> _number;

    mat.SetInt(_i, _j, _number);

    return istr;
}

ostream& operator<<(ostream& ostr, const TMatrix& mat)
{
    int n = mat.GetN();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i > j) ostr << 0;
            else ostr << mat.GetInt(i, j);
        }
        ostr << endl;
    }

    return ostr;
}


int main()
{
    int n;
    cin >> n;

    TMatrix a(n), b(n), c(n);
    a.SetInt(2, 2, 2);
    b.SetInt(5, 5, 5);
    cout << a;
    cout << b;
    //c = a + b;
    //cout << c;


    return 0;
}