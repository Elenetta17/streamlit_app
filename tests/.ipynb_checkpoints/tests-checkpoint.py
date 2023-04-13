import unittest
from dashboard import load_dataframe, load_shap_values
        
class TestLoadDataframe(unittest.TestCase):
    def test_df_size(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.shape, (48744, 45))
    def test_first_row_values(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        self.assertEqual(data.iloc[0,0], 378000.0)
        
# a completer
class TestLoadExplainer(unittest.TestCase):
    def test_explainer_expected_valur(self):
        data = load_dataframe('./test_kaggle_reduced.csv')
        explainer, shap_values = load_shap_values('explainer.pkl', data)


                
if __name__ == "__main__":
    unittest.main()
