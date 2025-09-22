import pandas as pd
from IPython import display

def df_display(df):
    with pd.option_context(
        'display.max_columns', None, 
        'display.max_rows', None,  
        'display.width', None,
        'display.max_colwidth', None,

    ):
        return (df
                  .style
                  .format(precision=2)
                  .set_properties(**{'text-align': 'left'})
                  .set_table_styles([{
                      'selector': 'th',
                      'props': [('text-align', 'left')]
                  }]))