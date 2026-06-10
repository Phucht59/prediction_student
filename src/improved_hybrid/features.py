import pandas as pd
import numpy as np
from typing import Tuple, List
from .config import DatasetSpec

def engineer_student_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = data.copy()
    
    # 1. Sequence features (for CNN-BiLSTM)
    sequence_cols = ['G1', 'G2']
    
    # 2. Engineered Context Features
    # Numeric features
    df['grade_delta'] = df['G2'] - df['G1']
    df['grade_mean'] = (df['G1'] + df['G2']) / 2
    df['grade_min'] = df[['G1', 'G2']].min(axis=1)
    df['grade_max'] = df[['G1', 'G2']].max(axis=1)
    df['grade_trend'] = np.sign(df['grade_delta'])
    
    df['fail_absence_risk'] = df['failures'] * df['absences']
    df['study_absence_ratio'] = df['studytime'] / (df['absences'] + 1)
    
    df['goout'] = pd.to_numeric(df['goout'], errors='coerce').fillna(0)
    df['Dalc'] = pd.to_numeric(df['Dalc'], errors='coerce').fillna(0)
    df['Walc'] = pd.to_numeric(df['Walc'], errors='coerce').fillna(0)
    df['social_alcohol_risk'] = df['goout'] + df['Dalc'] + df['Walc']
    
    # Map categorical yes/no to 1/0 for adding
    def yesno_to_int(col):
        return df[col].map({'yes': 1, 'no': 0}).fillna(0)
        
    df['support_count'] = yesno_to_int('schoolsup') + yesno_to_int('famsup') + yesno_to_int('paid')
    
    numeric_context_cols = [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
        'grade_delta', 'grade_mean', 'grade_min', 'grade_max', 'grade_trend',
        'fail_absence_risk', 'study_absence_ratio', 'social_alcohol_risk', 'support_count'
    ]
    
    categorical_context_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
        'nursery', 'higher', 'internet', 'romantic'
    ]
    
    # Ensure numeric types
    for col in numeric_context_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        
    return df, sequence_cols, numeric_context_cols, categorical_context_cols

def engineer_xapi_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = data.copy()
    
    # Resolve VisitedResources casing issue
    if 'VisITedResources' in df.columns:
        df.rename(columns={'VisITedResources': 'VisitedResources'}, inplace=True)
    if 'NationalITy' in df.columns:
        df.rename(columns={'NationalITy': 'Nationality'}, inplace=True)
        
    # Sequence pseudo-features (ordered by typical engagement depth)
    sequence_cols = ['AnnouncementsView', 'VisitedResources', 'raisedhands', 'Discussion']
    
    # Engineered Context Features
    for col in sequence_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['engagement_total'] = df['raisedhands'] + df['VisitedResources'] + df['AnnouncementsView'] + df['Discussion']
    df['resource_per_hand'] = df['VisitedResources'] / (df['raisedhands'] + 1)
    df['discussion_ratio'] = df['Discussion'] / (df['engagement_total'] + 1)
    
    df['absence_binary'] = df['StudentAbsenceDays'].map({'Under-7': 0, 'Above-7': 1}).fillna(0)
    
    numeric_context_cols = [
        'raisedhands', 'VisitedResources', 'AnnouncementsView', 'Discussion',
        'engagement_total', 'resource_per_hand', 'discussion_ratio', 'absence_binary'
    ]
    
    categorical_context_cols = [
        'gender', 'Nationality', 'PlaceofBirth', 'StageID', 'GradeID', 
        'SectionID', 'Topic', 'Semester', 'Relation',
        'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays'
    ]
    
    for col in numeric_context_cols:
        df[col] = df[col].astype(np.float32)
        
    return df, sequence_cols, numeric_context_cols, categorical_context_cols

def apply_feature_engineering(df: pd.DataFrame, spec: DatasetSpec):
    if spec.kind == 'student':
        return engineer_student_features(df)
    elif spec.kind == 'xapi':
        return engineer_xapi_features(df)
    else:
        raise ValueError(f"Unknown dataset kind: {spec.kind}")
