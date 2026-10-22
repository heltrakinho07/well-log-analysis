import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_modes_render_and_empty_batch_is_rejected(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app/streamlit_app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Previsor em Lote (CSV)').run()
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn('batch_result', app.session_state)
        app.sidebar.radio[0].set_value('Explorador de Dados').run()
        self.assertFalse(app.exception)

if __name__=='__main__': unittest.main()
