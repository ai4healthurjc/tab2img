import numpy as np
from numpy.random import choice, randint


def generate_swap_noise(m_features, p):
    arr = m_features.copy()
    n, m = arr.shape
    idx = range(n)
    swap_n = round(n * p)
    for i in range(m):
        col_vals = np.random.permutation(arr[:, i])  # change the order of the row
        swap_idx = np.random.choice(idx, size=swap_n)  # choose row
        arr[swap_idx, i] = np.random.choice(col_vals, size=swap_n)  # n*p row and change it
    return arr


def add_noise(tables, u, noise_model, margin=0):

    def add_noise_one_table(one_table, u, noise_model, margin=0):
        t_row, t_col = one_table.shape
        table_result = None

        if t_row > 1:
            col_prob_mat = get_prob_matrix(t_row, u, noise_model)

        if t_col > 1:
            row_prob_mat = get_prob_matrix(t_col, u, noise_model)

        if margin == 0:
            tables = []
            for x in range(1, t_row * t_col + 1):
                row = ((x - 1) % t_col) + 1
                col = ((x - 1) // t_col) + 1
                t_one_sample = np.zeros((t_row, t_col))

                if one_table[row - 1, col - 1] == 0:
                    tables.append(t_one_sample)
                    continue

                if t_row == 1 and t_col > 1:
                    prvector_row = row_prob_mat[:, col - 1]
                    row_index = choice(range(1, t_col + 1), size=one_table[row - 1, col - 1], p=prvector_row)
                    t_one_sample[row - 1, :] = np.bincount(row_index, minlength=t_col)[1:]
                    tables.append(t_one_sample)
                elif t_col == 1 and t_row > 1:
                    prvector_col = col_prob_mat[:, row - 1]
                    col_index = choice(range(1, t_row + 1), size=one_table[row - 1, col - 1], p=prvector_col)
                    t_one_sample[:, col - 1] = np.bincount(col_index, minlength=t_row)[1:]
                    tables.append(t_one_sample)
                elif t_row > 1 and t_col > 1:
                    valuevector_row = row_prob_mat[:, col - 1]
                    valuevector_col = col_prob_mat[:, row - 1]

                    col_index = np.searchsorted(valuevector_row, np.random.rand(one_table[row - 1, col - 1]))
                    row_index = np.searchsorted(valuevector_col, np.random.rand(one_table[row - 1, col - 1]))

                    for y in range(one_table[row - 1, col - 1]):
                        t_one_sample_tmp = np.zeros((t_row, t_col))
                        t_one_sample_tmp[row_index[y], col_index[y]] = 1
                        t_one_sample += t_one_sample_tmp

                    tables.append(t_one_sample)
                else:
                    raise ValueError("Wrong table size 1*1!")

            table_result = np.sum(tables, axis=0)
        elif margin == 1:
            table_result = np.transpose(np.apply_along_axis(lambda x: add_noise_one_table(x, u, noise_model, 0), 1, one_table))
        elif margin == 2:
            table_result = np.apply_along_axis(lambda x: add_noise_one_table(x, u, noise_model, 0), 0, one_table)
        else:
            raise ValueError("Wrong margin values! ([0,1,2])")

        return table_result

    def get_prob_matrix(base_num, u, noise_model):
        pr_vector = np.zeros((base_num, base_num))

        temp_sum = np.sum([np.abs(np.arange(1, base_num + 1) - x) for x in range(1, base_num + 1)], axis=0)

        for x in range(1, base_num + 1):
            a = np.arange(1, base_num + 1)
            res = (1 - np.abs(x - a) / temp_sum[x - 1]) * u / (base_num - 1)
            res[x - 1] += 1 - u
            pr_vector[x - 1, :] = res

        pr_vector = pr_vector * (1 - u) + u / base_num

        value_vector = np.zeros((base_num, base_num))
        for x in range(1, base_num + 1):
            if x == 1:
                value_vector[:, x - 1] = pr_vector[:, x - 1]
            else:
                value_vector[:, x - 1] = np.sum(pr_vector[:, :x], axis=1)

        return {"prvector": pr_vector, "valuevector": value_vector}

    def house_noise_model_prob_matrix(base_num, u):
        pr_vector = np.zeros((base_num, base_num))
        temp_sum = np.array([np.sum(np.abs(np.arange(1, base_num + 1) - x)) for x in range(1, base_num + 1)])

        for x in range(1, base_num + 1):
            res = (1 - np.abs(x - np.arange(1, base_num + 1)) / temp_sum[x - 1]) * u / (base_num - 1)
            res[x - 1] += 1 - u
            pr_vector[x - 1, :] = res

        pr_vector = pr_vector * (1 - u) + u / base_num

        value_vector = np.zeros((base_num, base_num))
        for x in range(1, base_num + 1):
            if x == 1:
                value_vector[:, x - 1] = pr_vector[:, x - 1]
            else:
                value_vector[:, x - 1] = np.sum(pr_vector[:, :x], axis=1)

        return {"prvector": pr_vector, "valuevector": value_vector}

    def candle_noise_model_prob_matrix(base_num, u):
        pr_vector = np.full((base_num, base_num), u / (base_num - 1))
        for i in range(base_num):
            pr_vector[i, i] = 1 - u

        value_vector = np.zeros((base_num, base_num))
        for x in range(1, base_num + 1):
            if x == 1:
                value_vector[:, x - 1] = pr_vector[:, x - 1]
            else:
                value_vector[:, x - 1] = np.sum(pr_vector[:, :x], axis=1)

        return {"prvector": pr_vector, "valuevector": value_vector}

    def house_noise_model_prob_matrix(base_num, u):
        return candle_noise_model_prob_matrix(base_num, u)

    if isinstance(tables, np.ndarray) or isinstance(tables, list):
        tables_noised = add_noise_one_table(np.array(tables), u, noise_model, margin)
    else:
        raise ValueError("Wrong input format!")

    return tables_noised


def add_house_noise(tables, u, margin=0):
    return add_noise(tables, u, "house", margin)


def add_candle_noise(tables, u, margin=0):
    return add_noise(tables, u, "candle", margin)


def test_noise_model_case(k=3):
    def test(noise_model, k):
        row_num = np.random.randint(2, 6, k)
        col_num = np.random.randint(2, 6, k)

        tables = [np.random.randint(1, 6, size=(row_num[i], col_num[i])) for i in range(k)]
        tables_XY = add_noise(tables, 0.5, noise_model, 0)
        tables_X = add_noise(tables, 0.5, noise_model, 1)
        tables_Y = add_noise(tables, 0.5, noise_model, 2)

        res = [np.sum(t) == np.sum(t_XY) and
               np.all(np.sum(t, axis=0) == np.sum(t_Y, axis=0)) and
               np.all(np.sum(t, axis=1) == np.sum(t_X, axis=1)) and
               len(set([len(set([t.shape[0], t_XY.shape[0], t_X.shape[0], t_Y.shape[0]])),
                        len(set([t.shape[1], t_XY.shape[1], t_X.shape[1], t_Y.shape[1]]))])) == 1
               for t, t_XY, t_X, t_Y in zip(tables, tables_XY, tables_X, tables_Y)]

        res = np.all(res)

        t = tables[0]
        t_XY = add_noise(t, 0.5, noise_model, 0)
        t_X = add_noise(t, 0.5, noise_model, 1)
        t_Y = add_noise(t, 0.5, noise_model, 2)

        res = np.append(res,
                        np.sum(t) == np.sum(t_XY) and
                        np.all(np.sum(t, axis=0) == np.sum(t_Y, axis=0)) and
                        np.all(np.sum(t, axis=1) == np.sum(t_X, axis=1)) and
                        len(set([len(set([t.shape[0], t_XY.shape[0], t_X.shape[0], t_Y.shape[0]])),
                                 len(set([t.shape[1], t_XY.shape[1], t_X.shape[1], t_Y.shape[1]]))])) == 1)

        if np.all(res):
            print(f"All test cases passed for noise_model: \"{noise_model}\"!")
        else:
            print(f"{np.sum(res)} / {len(res)} test cases passed for noise_model: \"{noise_model}\"!")

    test("house", k)
    test("candle", k)

# Test the random k 5*5 cases
test_noise_model_case(5)
