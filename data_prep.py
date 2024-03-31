
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

small_value = 1e-10  # Small constant to avoid division by zero

# # Load spreadsheet
# xl = pd.ExcelFile('assessment.xlsx')

# # Load a sheet into a DataFrame by name
# df = xl.parse('Data')

def fill_nan_with_ml(df: pd.DataFrame, target: str, features: list) -> pd.DataFrame:
    # Step 1: Prepare the data
    train_data = df[df[target].notnull()]
    test_data = df[df[target].isnull()]

    # Step 2: Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_data[features], train_data[target])

    # Step 3: Predict the missing values
    predicted_values = model.predict(test_data[features])

    # Step 4: Fill the missing values in the original DataFrame
    df.loc[df[target].isnull(), target] = predicted_values

    return df

def datatype_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df_cp = df.copy(deep=True)
    # List of columns to fill NaN and convert to int
    columns = ['INCM_TYP', 'CASATD_CNT', 'PC', 'MTHTD','Asset value','MTHCASA','UT_AVE','N_FUNDS', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_AVE', 'MIN_MTH_TRN_AMT','MAX_MTH_TRN_AMT','CC_LMT','HL_tag','AL_tag','pur_price_avg','AVG_TRN_AMT','DRvCR']

    # Apply fillna and astype to each column
    df_cp[columns] = df_cp[columns].fillna(0).astype(int)

    # Convert C_seg to numerical value using Label Encoding
    df_cp['C_seg'] = df_cp['C_seg'].astype('category').cat.codes #0 =Affluent, 1 = Normal

    return df_cp

def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df_cp= df.copy(deep=True)
    # def calculate_casastd_cnt_unq(row):
    #     """
    #     Calculates the unique count of CASATD_CNT based on the given row.

    #     Parameters:
    #     - row: A dictionary representing a row of data.

    #     Returns:
    #     - The unique count of CASATD_CNT.

    #     Note:
    #     - If CASATD_CNT is 1, the function returns 1.
    #     - If CASATD_CNT is 2, the function returns 2.
    #     - For any other value of CASATD_CNT, the function returns 2.
    #     """
    #     if row['CASATD_CNT'] == 1:
    #         return 1
    #     elif row['CASATD_CNT'] == 2:
    #         return 2
    #     else:
    #         return 2

    # df_cp['CASASTD_CNT_UNQ'] = df_cp.apply(calculate_casastd_cnt_unq, axis=1)
    # # df[['C_ID', 'CASATD_CNT', 'NUM_PRD', 'CASASTD_CNT_UNQ']].head()

    # if MAXTD or MTHTD have some value create new column with name IS_TD with value 1 else 0
    df_cp['IS_TD'] = np.where((df_cp['MAXTD'].notnull() | df_cp['MTHTD'].notnull()), 1, 0)
    # df[['C_ID', 'CASATD_CNT', 'NUM_PRD','MTHTD','MAXTD' ,'IS_TD']].head()

    # if MTHCASA or MAXCASA have some value create new column with name IS_CASA with value 1 else 0
    df_cp['IS_CASA'] = np.where((df_cp['MTHCASA'].notnull() | df_cp['MAXCASA'].notnull()), 1, 0)
    # df[['C_ID', 'CASATD_CNT', 'NUM_PRD','MTHCASA','MAXCASA' ,'IS_CASA']].head()

    # if N_FUNDS or MAXUT have some count then create new column with name IS_FUNDS with value 1 else 0
    df_cp['IS_FUNDS'] = np.where((df_cp['N_FUNDS'].notnull() | df_cp['MAXUT'].notnull()), 1, 0)
    # df[['C_ID','NUM_PRD','UT_AVE','N_FUNDS','MAXUT' ,'IS_FUNDS']].head()

    # IF CC_CHECK is 1 it means these records customer who don't have any outstanding balancec every month, so regular payee
    # create new column named CC_CHECK with value 1 if above line return positive result else 0
    # df['CC_CHECK'] = np.where(((df['CC_AVE'].isnull()| df['CC_AVE']==0) & (df['MAX_MTH_TRN_AMT'] + df['MIN_MTH_TRN_AMT']+ df['AVG_TRN_AMT']+ df['ANN_TRN_AMT']+ df['ANN_N_TRX']+ df['CC_LMT']) > 0), 1, 0)
    # df[(df['CC_AVE'].isnull()| df['CC_AVE']==0) & (df['MAX_MTH_TRN_AMT'] + df['MIN_MTH_TRN_AMT']+ df['AVG_TRN_AMT']+ df['ANN_TRN_AMT']+ df['ANN_N_TRX']+ df['CC_LMT']) > 0][['CC_AVE','MAX_MTH_TRN_AMT', 'MIN_MTH_TRN_AMT', 'AVG_TRN_AMT', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_LMT','CC_CHECK']].head()

    # change NaN to 0 for these columns 'CC_AVE','MAX_MTH_TRN_AMT', 'MIN_MTH_TRN_AMT', 'AVG_TRN_AMT', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_LMT'
    df_cp[['CC_AVE','MAX_MTH_TRN_AMT', 'MIN_MTH_TRN_AMT', 'AVG_TRN_AMT', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_LMT']] = df_cp[['CC_AVE','MAX_MTH_TRN_AMT', 'MIN_MTH_TRN_AMT', 'AVG_TRN_AMT', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_LMT']].fillna(0)

    # if sum of above columns is 0 or NULL then create new column with name IS_CC_ZERO with value o else 1
    df_cp['IS_CC'] = np.where(((df_cp['CC_AVE'] + df_cp['MAX_MTH_TRN_AMT'] + df_cp['MIN_MTH_TRN_AMT'] + df_cp['AVG_TRN_AMT'] + df_cp['ANN_TRN_AMT'] + df_cp['ANN_N_TRX'] + df_cp['CC_LMT']).isnull() | (df_cp['CC_AVE'] + df_cp['MAX_MTH_TRN_AMT'] + df_cp['MIN_MTH_TRN_AMT'] + df_cp['AVG_TRN_AMT'] + df_cp['ANN_TRN_AMT'] + df_cp['ANN_N_TRX'] + df_cp['CC_LMT'])==0), 0, 1)
    # df[['C_ID', 'CC_AVE', 'MAX_MTH_TRN_AMT', 'MIN_MTH_TRN_AMT', 'AVG_TRN_AMT', 'ANN_TRN_AMT', 'ANN_N_TRX', 'CC_LMT', 'IS_CC', 'CC_CHECK']].head()

    df_cp['IS_LOAN']= np.where((df_cp['HL_tag'] + df_cp['AL_tag'])>0, 1, 0)  

    # show records where (df['HL_tag'] + df['AL_tag'])>1 
    # df[(df['HL_tag'] + df['AL_tag'])>1][['C_ID','HL_tag','AL_tag','IS_LOAN']].head()
    return df_cp[['IS_TD','IS_CASA','IS_FUNDS','IS_CC','IS_LOAN']]

####--------------------------------------####

# Derive more features
def derive_features(df: pd.DataFrame) -> pd.DataFrame:

    # Education Level Encoding
    edu_mapping = {
    "Below O-Levels": 0,
    "O-Levels": 0,
    "A-Levels": 0,
    "Diploma": 1,
    "Degree": 2,
    "Technical/Vocational Qualifications": 1,
    "Professional Qualifications": 2,
    "Masters": 3,
    "PHD/Doctorate": 3,
    "Others": 0 ,
    np.nan: 0}
    
    df['C_EDU_Encoded'] = df['C_EDU'].map(edu_mapping)


    # House Type Encoding
    # df['C_HSE'].unique()
    house_mapping = {
    "HDB 1-3 ROOM": 0,
    "HDB 4-5 ROOM": 1,
    "HDB EXECUTIVE APARTMENT/ MANSIONETTE": 1,
    "EXECUTIVE CONDOMINIUM": 2,
    "PRIVATE APARTMENT": 2,
    "PRIVATE CONDOMINIUM": 2,
    "SEMI-DETACHED": 3,
    "TERRACE": 3,
    "BUNGALOW": 4,
    "SHOPHOUSE": 4,
    "INDUSTRIAL BUILDING": 5,
    "COMMERICAL BUILDING": 5,
    "OFFICE": 5,
    "HOTEL/ SERVICE APARTMENT": 5,
    "Others": 0,
    np.nan: 0 }

    df['C_HSE_Encoded'] = df['C_HSE'].map(house_mapping)
    # df[['C_HSE_Encoded','C_HSE']]

    #Occupation type encoding
    # df['gn_occ'].unique()
    occ_mapping = {
    "STUDENT": 0,
    "BLUE COLLAR": 1,
    "HOUSEWIFE": 0,
    "OTHERS": 0,
    "PMEB": 2,
    "WHITE COLLAR": 2,
    "RETIREE": 1,
    "Others": 0,
    np.nan: 0 }

    df['C_OCC_Encoded'] = df['gn_occ'].map(occ_mapping)

    # Age Group
    # Fill NaN values in 'C_AGE' using machine learning
    Age_has_nan = df['C_AGE'].isnull().any()
    if Age_has_nan:
        df = fill_nan_with_ml(df, 'C_AGE', ['C_EDU_Encoded', 'C_HSE_Encoded', 'INCM_TYP', 'C_OCC_Encoded', 'NUM_PRD', 'Asset value'])

    min_age = df['C_AGE'].min()
    max_age = df['C_AGE'].max()
    # Create bins
    bins = pd.cut(df['C_AGE'], bins=[min_age, 18, 30, 45, 60, max_age], labels=[0, 1, 2, 3, 4], include_lowest=True, right=False)
    # Assign bins to new column
    df['Age_Group'] = bins
    # convert Age_Group to int and fill NaN values with 0
    df['Age_Group'] = df['Age_Group'].fillna(0).astype(int)
    
    # Total Number of Accounts
    # df['Total_Accounts'] = df['CASATD_CNT'] + df['NUM_PRD']

    # Total CASA Balance
    df['Total_CASA_Balance'] = df['CASATD_CNT'] * df['MTHCASA']+ small_value

    # Total TD Balance
    df['Total_TD_Balance'] = df['NUM_PRD'] * df['MTHTD']

    # # Total Credit Amount
    # df['Total_Credit_Amount'] = df['MAX_MTH_TRN_AMT'] * 12

    # # Total Debit Amount
    # df['Total_Debit_Amount'] = df['MIN_MTH_TRN_AMT'] * 12

    # # Total Assets to Income Ratio
    # df['Assets_to_Income_Ratio'] = df['Asset value'] / (df['INCM_TYP'] * 1000)  # Assuming income is in thousands

    # Loan Utilization Ratio
    # loan value utilised for credit limit
    # df['Loan_Utilization_Ratio'] = (df['HL_tag'] + df['AL_tag']) / df['CC_LMT']

    # Average Transaction Amount per Account
    # df['Avg_Transaction_Amount_per_Account'] = df['ANN_TRN_AMT'] / df['Total_Accounts']

    # Average CC transaction size
    df['Avg_CC_Transaction_Size'] = df['ANN_TRN_AMT'] / (df['ANN_N_TRX']+small_value)
    # df[['Avg_CC_Transaction_Size','ANN_TRN_AMT','ANN_N_TRX']]

    # CC Outstanding to Credit Limit Ratio
    df['CC_Outstanding_to_Credit_Limit_Ratio'] = round((df['CC_AVE'] / (df['CC_LMT']+small_value))*100,2)
    # df[['CC_AVE','CC_LMT','CC_Outstanding_to_Credit_Limit_Ratio']]

    # Assets to Product Ratio
    df['Assets_to_Product_Ratio'] = (df['Asset value'] / (df['NUM_PRD'] + small_value)).round(2)
    # df[['Asset value','NUM_PRD','Assets_to_Product_Ratio']]


    # Allocation of Asset for diffrent products as ratio
    # For CASATD
    df['CASATD_Asset_Ratio'] = df['CASATD_CNT']*(df['MTHCASA'] + df['MTHTD'])/(df['Asset value'] + small_value)
    df[['CASATD_CNT','MTHCASA','MTHTD','Asset value','CASATD_Asset_Ratio']]

    # For UT
    df['UT_Asset_Ratio'] = df['N_FUNDS']*(df['UT_AVE'])/(df['Asset value']+ small_value)
    df[['N_FUNDS','UT_AVE','Asset value','UT_Asset_Ratio']]

    # For CC
    df['CC_Asset_Ratio'] = (df['ANN_TRN_AMT'] / (df['Asset value'] + small_value)).round(2)
    # df[['ANN_TRN_AMT','Asset value','CC_Asset_Ratio']]

    # Loan purchase price to asset ratio
    df['Loan_to_Asset_Ratio'] = (df['pur_price_avg'] / (df['Asset value'] + small_value)).round(2)
    # df[['pur_price_avg','Asset value','Loan_to_Asset_Ratio']]
    return df


def standardize_columns(df: pd.DataFrame, column_list: list) -> pd.DataFrame:
    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    for column in column_list:
        # Apply log transformation to handle skewness
        df_copy[column] = np.log1p(df_copy[column])

        # Standardize the column using RobustScaler
        df_copy[column] = scaler.fit_transform(df_copy[column].values.reshape(-1, 1))

    return df_copy

def data_processing(filename='assessment.xlsx',sheetname='Data'):
    # Load spreadsheet
    try:
        xl = pd.ExcelFile(filename)
    except:
        return "File not found"
    try:
        # Load a sheet into a DataFrame by name
        df = xl.parse(sheetname)
    except:
        return "Sheet not found"
    
    df = datatype_cleanup(df)
    df = create_product_features(df)
    df = derive_features(df)
    
    final_columns = ['C_ID','IS_TD','IS_CASA','IS_FUNDS','IS_CC','IS_LOAN','Age_Group','C_EDU_Encoded','C_HSE_Encoded','C_OCC_Encoded','Total_CASA_Balance','Total_TD_Balance','Avg_CC_Transaction_Size','CC_Outstanding_to_Credit_Limit_Ratio','Assets_to_Product_Ratio','CASATD_Asset_Ratio','UT_Asset_Ratio','CC_Asset_Ratio','Loan_to_Asset_Ratio','NUM_PRD','C_seg']

    column_list= ['Total_CASA_Balance','Total_TD_Balance','Avg_CC_Transaction_Size','CC_Outstanding_to_Credit_Limit_Ratio','Assets_to_Product_Ratio','CASATD_Asset_Ratio','UT_Asset_Ratio','CC_Asset_Ratio','Loan_to_Asset_Ratio']
    df_standardized = standardize_columns(df, column_list)
    return df_standardized[final_columns]

if __name__ == '__main__':
    filename = 'assessment.xlsx'
    sheetname = 'Data'
    print(data_processing(filename,sheetname))
