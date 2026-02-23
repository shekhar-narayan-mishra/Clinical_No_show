import pandas as pd

def preprocess(df):

    df = df.copy()

    df.rename(columns={'No-show': 'No_show'}, inplace=True)

    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    df['lead_time'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['appointment_dayofweek'] = df['AppointmentDay'].dt.dayofweek

    df['Gender'] = df['Gender'].map({'F':0, 'M':1})

    df['age_group'] = pd.cut(
        df['Age'],
        bins=[-1,12,18,35,60,200],
        labels=[0,1,2,3,4],
        include_lowest=True
    ).astype(int)

    df['lead_time_bucket'] = pd.cut(
        df['lead_time'],
        bins=[-1000,0,3,7,30,1000],
        labels=[0,1,2,3,4],
        include_lowest=True
    ).astype(int)

    df['sms_lead_interaction'] = df['SMS_received'] * df['lead_time']
    df['is_weekend'] = df['appointment_dayofweek'].isin([5,6]).astype(int)

    df.drop([
        'PatientId',
        'AppointmentID',
        'ScheduledDay',
        'AppointmentDay',
        'Neighbourhood',
        'No_show'
    ], axis=1, inplace=True, errors='ignore')

    return df
