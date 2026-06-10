import pandas as pd
import numpy as np

def engineer_student_features(df: pd.DataFrame):
    """Sequence: G1, G2. Context: rest."""
    seq_cols = ['G1', 'G2']
    
    # Optional engineered features
    df['grade_delta'] = df['G2'] - df['G1']
    df['grade_mean'] = (df['G1'] + df['G2']) / 2
    df['fail_absence_risk'] = df['failures'] * df['absences']
    
    # Categoricals
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                'nursery', 'higher', 'internet', 'romantic']
    
    # Numerics
    num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                'grade_delta', 'grade_mean', 'fail_absence_risk']
                
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        
    return df, seq_cols, num_cols, cat_cols

def engineer_xapi_features(df: pd.DataFrame):
    if 'VisITedResources' in df.columns:
        df.rename(columns={'VisITedResources': 'VisitedResources'}, inplace=True)
    if 'NationalITy' in df.columns:
        df.rename(columns={'NationalITy': 'Nationality'}, inplace=True)
        
    seq_cols = ['AnnouncementsView', 'VisitedResources', 'raisedhands', 'Discussion']
    
    for col in seq_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        
    df['engagement_total'] = df['raisedhands'] + df['VisitedResources'] + df['AnnouncementsView'] + df['Discussion']
    df['absence_binary'] = df['StudentAbsenceDays'].map({'Under-7': 0, 'Above-7': 1}).fillna(0)
    
    num_cols = ['raisedhands', 'VisitedResources', 'AnnouncementsView', 'Discussion',
                'engagement_total', 'absence_binary']
                
    cat_cols = ['gender', 'Nationality', 'PlaceofBirth', 'StageID', 'GradeID', 
                'SectionID', 'Topic', 'Semester', 'Relation',
                'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']
                
    for col in num_cols:
        df[col] = df[col].astype(np.float32)
        
    return df, seq_cols, num_cols, cat_cols

def apply_feature_engineering(df: pd.DataFrame, kind: str):
    if kind == 'student':
        return engineer_student_features(df)
    else:
        return engineer_xapi_features(df)
