import numpy as np
import pandas as pd
from shift_adapt import ShiftAdapt

def csv_to_df(file_path):
    return pd.read_csv(file_path)

def main():
    csv_file_path = 'train-TX.csv'
    train_df = csv_to_df(csv_file_path)
    val_df = csv_to_df('val-TX.csv')
    test1 = csv_to_df('test1-TX.csv')
    test2 = csv_to_df('test2-FL.csv')
    test3 = csv_to_df('test3-FL.csv')
    columns_to_keep = ['Distance(mi)',
                       'Temperature(F)',
                       'Wind_Chill(F)',
                       'Humidity(%)',
                       'Pressure(in)',
                       'Visibility(mi)',
                       'Wind_Speed(mph)',
                       'Precipitation(in)',
                       'Amenity',
                       'Bump',
                       'Crossing',
                       'Give_Way',
                       'Junction',
                       'No_Exit',
                       'Railway',
                       'Roundabout',
                       'Station',
                       'Stop',
                       'Traffic_Calming',
                       'Traffic_Signal',
                       'Turning_Loop',
                       'Severity' ]
    train_df = train_df[columns_to_keep]
    val_df = val_df[columns_to_keep]
    test1 = test1[columns_to_keep]
    test2 = test2[columns_to_keep]
    test3 = test3[columns_to_keep]
    expt_instance = ShiftAdapt(train_df, val_df, test1, test2, test3)
    expt_instance.train()
    val_acc = expt_instance.valid()
    test1_acc = expt_instance.test1()
    test2_acc = expt_instance.test2()
    test3_acc = expt_instance.test3()
    weight1, accuracy1, probability1 = expt_instance.label_shift_1()
    weight2, accuracy2, probability2 = expt_instance.label_shift_2()
    weight3, accuracy3, probability3 = expt_instance.label_shift_3()

    keys_order = ['rf', '3nn', '9nn', 'gpc', 'd_freq', 'd_strat']
    cols_order = ['val','test1','test2','test3']

    table = [[None] * 4 for _ in range(0,6)]
    for i, key in enumerate(val_acc):
        table[i][0] = val_acc[key]
    
    for i, key in enumerate(keys_order):
        if key in test1_acc and key in accuracy1:
            table[i][1] = (test1_acc[key], accuracy1[key])
        elif key in test1_acc:
            table[i][1] = test1_acc[key]
        elif key in accuracy1:
            table[i][1] = accuracy1[key]

    for i, key in enumerate(keys_order):
        if key in test2_acc and key in accuracy2:
            table[i][2] = (test2_acc[key], accuracy2[key])
        elif key in test2_acc:
            table[i][2] = test2_acc[key]
        elif key in accuracy2:
            table[i][2] = accuracy2[key]

    for i, key in enumerate(keys_order):
        if key in test3_acc and key in accuracy3:
            table[i][3] = (test3_acc[key], accuracy3[key])
        elif key in test3_acc:
            table[i][3] = test3_acc[key]
        elif key in accuracy3:
            table[i][3] = accuracy3[key]

    max_row_label_length = max(len(label) for label in keys_order)

    print("{:<{width}}".format("", width=max_row_label_length), end="")
    print("{:<12}".format("val"), end="")
    for col_label in cols_order[1:]:
        print("{:<13}".format(col_label), end="")
    print()

    for i, row_label in enumerate(keys_order):
        print(f"{row_label: <{max_row_label_length}}", end="")
        for j in range(len(cols_order)):
            if isinstance(table[i][j], tuple):
                print(f"({table[i][j][0]:.2f}, {table[i][j][1]:.2f})", end=" "*(13 - len(str(table[i][j]))))
            else:
                print(f"{table[i][j]:.2f} ", end=" "*(13 - len(str(table[i][j]))))
        print()

    key_order = ['rf', '3nn', '9nn', 'gpc']
    col_labels = ['test1', 'test2', 'test3']

    weights_table = [[None] * 3 for _ in range(0,4)]
    for i, key in enumerate(key_order):
        weights_table[i][0] = weight1[key]
        weights_table[i][1] = weight2[key]
        weights_table[i][2] = weight3[key]

    rounded_data = [[[[round(num, 2) for num in inner_array] if isinstance(inner_array, np.ndarray) else round(inner_array, 2) 
                      for inner_array in sublist] for sublist in sublist2] for sublist2 in weights_table]
    
    print(" " * 7, end="")
    for col_label in col_labels:
        print(f"{col_label: <25}", end="")
    print()

    for i, row_label in enumerate(key_order):
        print(f"{row_label: <7}", end="")
        for j in range(len(col_labels)):
            print(f"{rounded_data[i][j]} ", end="")
        print()



if __name__ =="__main__":
    main()
