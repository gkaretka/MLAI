def sum_elements(arr):
    table = ["0", "1", "00", "11", "000", "111", "0000", "1111", "00000", "11111"]
    mapa = {}
    cnt = 1
    char = arr[0]
    for i in range(1, len(arr)):
        if arr[i] == char:
            cnt = cnt + 1
        else:
            if char == 0:
                idx = (cnt-1)*2
            else:
                idx = (cnt-1)*2 + 1

            val = int(mapa.get(table[idx], 0))
            val = val + 1
            mapa.update({str(table[idx]): int(val)})

            # reset
            cnt = 1
            char = arr[i]

    s_mapa = sorted(list(mapa.items()), key=lambda key: len(key[0]))
    for element in s_mapa:
        print(element)
