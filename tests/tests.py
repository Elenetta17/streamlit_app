import unittest
from dashboard import load_dataframe, load_shap_values
        
class TestLoadDataframe(unittest.TestCase):
    def test_df_size(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.shape, (48744, 32))
    def test_first_row_values(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.iloc[0,0], -19241.0)
        
# a completer
class TestLoadExplainer(unittest.TestCase):
    def test_explainer_expected_value(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        explainer, shap_values = load_shap_values('explainer.pkl', data)
        expected_value = [0.6200196909299585, -0.6200196909299585]
        self.assertEqual(expected_value, explainer.expected_value)
    def test_explainer_feature_names(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        explainer, shap_values = load_shap_values('explainer.pkl', data)
        features_names = ['REGION_POPULATION_RELATIVE',
 'DAYS_EMPLOYED',
 'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
 'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
 'NAME_EDUCATION_TYPE_Higher education',
 'PREV_NAME_PORTFOLIO_POS_MEAN',
 'POS_MONTHS_BALANCE_MAX',
 'BURO_DAYS_CREDIT_ENDDATE_MAX',
 'BURO_DAYS_CREDIT_MAX',
 'PREV_CNT_PAYMENT_MEAN',
 'NAME_EDUCATION_TYPE_Secondary / secondary special',
 'APPROVED_RATE_DOWN_PAYMENT_MAX',
 'DAYS_REGISTRATION',
 'BURO_DAYS_CREDIT_MIN',
 'PREV_NAME_CLIENT_TYPE_New_MEAN',
 'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
 'PAYMENT_RATE',
 'APPROVED_DAYS_DECISION_MIN',
 'FLAG_DOCUMENT_3',
 'PREV_NAME_PORTFOLIO_Cash_MEAN',
 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
 'EXT_SOURCE_3',
 'EXT_SOURCE_2',
 'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',
 'PREV_NAME_YIELD_GROUP_middle_MEAN',
 'APPROVED_DAYS_DECISION_MAX',
 'DAYS_BIRTH',
 'PREV_DAYS_DECISION_MAX',
 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
 'INSTAL_COUNT',
 'PREV_NAME_YIELD_GROUP_high_MEAN',
 'AMT_CREDIT']
        self.assertEqual(features_names, explainer.feature_names)


                
if __name__ == "__main__":
    unittest.main()
