class SimplexMethod:
    def __init__(self, z, eq, b):
        (z, eq, b)   = map(lambda t: np.array(t), [z, eq, b])
        self.extra, self.table = self.tableFormation(eq, b, z)
        x, y = self.solve(self.table, eq, z)
        print(f'Optimal solution\n Z = {-y}')
        print('Optimum Values:')
        for i in x:
            print(i)

    def tableFormation(self, eq, b, z):
        m, n = eq.shape
        rng, tabinitial = self.initiateTable(eq, b, z)
        identity = np.vstack((np.identity(m), np.zeros(m)))
        return rng, np.concatenate((tabinitial, identity), axis=1)

    def initiateTable(self, eq, b, z):
        [m, n] = eq.shape
        if m != b.size or n != z.size:
            print('Error')
            exit(1)
        result = np.column_stack((eq, b))
        result = np.append(result, np.column_stack((z, 0)), axis=0)
        return range(n, n + m), result

    def positive(self, v):
        return all(v >= 0), np.amin(v), np.where(v == np.amin(v))
    
    def smallIndex(self, v):
        return np.where(v > 0, v, np.inf).argmin()

    def minn(self, a, b, m):
        out = []
        for i in range(0, m-1):
            if b[i] != 0:
                out.append(a[i] / b[i])             
        return self.smallIndex(np.array(out))

    def solve(self, tab, A, c):
        opt, tab_b, tab_c, tab_A    = self.start(tab, A)
        m, n                        = tab.shape
        sign, minimum, index_min    = self.positive(tab_c)
        if (index_min[0] > c.size).all():
            index_min = list(index_min)
            index_min[0] += 1
            index_min = tuple(index_min)
        tab = tab.astype(np.float32)
        if sign:
            return tab_b, opt
        else:
            if all(tab[:,index_min] <= 0):
                print('Error')
                exit(1)
            else:
                A_s = tab[:A.shape[0],index_min]
                index_pivot = self.minn(tab_b, A_s, m)
                ligne_pivot = tab[index_pivot]
                colonne_pivot = tab[:,index_min]
                pivot = tab[index_pivot,index_min]
                tab[index_pivot] = ligne_pivot / float(pivot)
                for i in range(0, len(tab)):
                    if not np.array_equal(tab[i], tab[index_pivot]):
                        tab[i] = tab[i] - tab[index_pivot] * tab[i, index_min]
                return self.solve(tab, A, c)
    
    def start(self, tab, A):
        m, n    = A.shape
        opt = -tab[m, n]
        tab_b = tab[:m, n]
        tab_c = np.concatenate((tab[m , 0:n], tab[m , n + 1:]))
        tab_A = np.hstack((tab[0:m ,0:n ], tab[ 0:m , n + 1 :]))
        return opt, tab_b, tab_c, tab_A

if __name__ == '__main__':
    import numpy as np
    from pandas import DataFrame as pd
    import math

    # Contants:b
    b = [24,6,1,2]

    # Objective Function:z
    z = [1,-5,-4]

    # Equations:eq
    eq = [[0,6,4],[0,1,2],[0,-1,1],[0,0,1]]
    b = SimplexMethod([z],eq,b)