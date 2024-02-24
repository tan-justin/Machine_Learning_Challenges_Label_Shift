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


if __name__ =="__main__":
    main()
