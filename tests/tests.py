import unittest
from dashboard import load_dataframe, load_shap_values

# check loading correct dataframe        
class TestLoadDataframe(unittest.TestCase):
    def test_df_size(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.shape, (48744, 32))
    def test_first_row_value(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.iloc[0, 0], 1778.0)
        
        
# check loading correct dataframe        
class TestLoadExplainer(unittest.TestCase):
    def test_explainer_expected_value(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        explainer, shap_values = load_shap_values('explainer.pkl', data)
        expected_value = [0.46541173759480076, -0.46541173759480076]
        self.assertEqual(expected_value, explainer.expected_value)
    def test_explainer_feature_names(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        explainer, shap_values = load_shap_values('explainer.pkl', data)
        features_names = [
            'BURO_DAYS_CREDIT_ENDDATE_MAX',
            'PAYMENT_RATE',
            'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
            'DAYS_EMPLOYED',
            'PREV_NAME_CLIENT_TYPE_New_MEAN',
            'APPROVED_DAYS_DECISION_MIN',
            'FLAG_DOCUMENT_3',
            'AMT_CREDIT',
            'PREV_NAME_YIELD_GROUP_high_MEAN',
            'BURO_DAYS_CREDIT_MIN',
            'INSTAL_COUNT',
            'APPROVED_RATE_DOWN_PAYMENT_MAX',
            'APPROVED_DAYS_DECISION_MAX',
            'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
            'BURO_DAYS_CREDIT_MAX',
            'POS_MONTHS_BALANCE_MAX',
            'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',
            'PREV_NAME_YIELD_GROUP_middle_MEAN',
            'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
            'EXT_SOURCE_3',
            'DAYS_REGISTRATION',
            'DAYS_BIRTH',
            'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
            'NAME_EDUCATION_TYPE_Secondary / secondary special',
            'REGION_POPULATION_RELATIVE',
            'NAME_EDUCATION_TYPE_Higher education',
            'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
            'EXT_SOURCE_2',
            'PREV_NAME_PORTFOLIO_POS_MEAN',
            'PREV_CNT_PAYMENT_MEAN',
            'PREV_DAYS_DECISION_MAX',
            'PREV_NAME_PORTFOLIO_Cash_MEAN']
        self.assertEqual(features_names, explainer.feature_names)

                
if __name__ == "__main__":
    unittest.main()
