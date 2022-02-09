import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as path_effects
# from matplotlib.pyplot import figure
import plotly.express as px


st.set_page_config(page_title="Finance Dashboard",
                  page_icon=":bar_chart:",
                  layout="wide"
                  )

st.markdown("""
<style>
.big-font {
    font-size:14px !important;
}
</style>
""", unsafe_allow_html=True)




df_needed_abs = pd.read_csv('OneDrive/Desktop/Data_Science/14_FINAL_PROJECT/df_needed_abs.csv')
# ================================= This contains all the entries ==========================================
lean_exp = pd.read_csv('OneDrive/Desktop/Data_Science/14_FINAL_PROJECT/lean_exped_clean.csv')
# ================================= This contains only monthly entries ==========================================
monthly_expenses_per_cat = pd.read_csv('OneDrive/Desktop/Data_Science/14_FINAL_PROJECT/Final_clean.csv')


ls_year = monthly_expenses_per_cat['year'].unique().tolist()
ls_year.sort()
ls_month =monthly_expenses_per_cat['month'].unique().tolist()
ls_month.sort()


# monthly_expenses_per_cat = (
# lean_exp
#     .groupby(['year','month','category','avg_salary','monthly_expenses'])
#     .agg({'Amount':'sum'})
#     .sort_values(['year','month'])
#     .assign(rate = (lambda x: round(x.Amount/avg_revenue, 2).abs()))
#     .reset_index()
# )
st.sidebar.header("Please Filter Here:")
Category = st.sidebar.multiselect(
    "Select category:",
    options=lean_exp["category"].unique(),
    default=lean_exp["category"].unique()[0]
)

Month = st.sidebar.multiselect(
    "Select month:",
    options=ls_month,
   default=ls_month[8],
)

Year = st.sidebar.multiselect(
    "Select year:",
    options=lean_exp["year"].unique(),
    default=lean_exp["year"].unique()[0]
)

df_selection = lean_exp.query(
    "category == @Category & year == @Year & month ==@Month"
)


df_selection_exp = lean_exp.query(
    "Amount < 0 & category ==@Category & month ==@Month & year ==@Year"
)

df_selection_rev = lean_exp.query(
    "avg_salary > 0 & category ==@Category & month ==@Month & year ==@Year"
)

st.title(":bar_chart: Finance Dashboard")
# ================================= Title of the website ==========================================

# ================================= Header ========================================================
st.write("""
### Analysis
""")


# sales_by_product_line = (
#     df_selection.groupby(by=["category"]).sum()[["Amount"]].sort_values(by="Amount")
# )
    
# fig_product_sales = px.bar(
#     sales_by_product_line,
#     x="Amount",
#     y=sales_by_product_line.index,
#     orientation="h",
#     title="<b>Sales by Product Line</b>",
#     color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
#     template="plotly_white",
# )
        

# fig_product_sales.update_layout(
#     plot_bgcolor="rgba(0,0,0,0)",
#     xaxis=(dict(showgrid=False))
# )
# ================ Avg Income per month from all monthly sal entries ==============================
avg_revenue = round(lean_exp.avg_salary.mean(),2)


# ================ Apply Avg Income per month in a list ===========================================
avg_revenue_list = [avg_revenue for i in range(len(monthly_expenses_per_cat))]
monthly_expenditures_list = (monthly_expenses_per_cat['monthly_expenses']).abs().unique().tolist()

years_months = monthly_expenses_per_cat.copy()
years_months['month_year'] = years_months['year'].astype(str)+'-'+years_months['month'].astype(str)
# ================ Apply Months in a list =========================================================
month_year_ls = years_months.month_year.unique().tolist()

# ================ Apply Months_Years  ============================================================
monthly_expenses_per_cat['month_year'] = monthly_expenses_per_cat['year'].astype(str)+' '+monthly_expenses_per_cat['month'].astype(str)
monthly_expenses_per_cat.rate =monthly_expenses_per_cat.apply(lambda row: round(row.rate,2),axis=1)

# ================ Code to fancy red/green chart ===================================================
import plotly.offline as pyo
import plotly.graph_objs as go
# pyo.init_notebook_mode()

# ================ checkbox to display the dataframe ===============================================
temp = monthly_expenses_per_cat.loc[:,monthly_expenses_per_cat.columns.isin(['year','month','category','rate'])]
temp = temp.rename(columns={'rate' : 'mon_exp/income'})



if st.checkbox('Display Transactions'):
#     st.dataframe(df_selection)
    df_selection['avg_salary'] = df_selection.apply(lambda row: avg_revenue, axis=1)
    st.dataframe(df_selection.style.format(subset=['Amount', 'avg_salary','monthly_expenses'], formatter="{:.2f}"))
                                           
mon_year, rev_numbers, exp_numbers = month_year_ls, avg_revenue_list, monthly_expenditures_list
## calculate percentage change of an animal from SF to LA
exp_rev_percent_change = [100*(rev_count - exp_count) / rev_count 
                        for rev_count, exp_count in zip(rev_numbers, exp_numbers)]

# pass the text to the second graph_object only
fig = go.Figure(data=[
    go.Bar(name='Revenues', x=mon_year, y=rev_numbers, marker_color='darkgreen'),
    go.Bar(name='Expenditures', x=mon_year, y=exp_numbers, marker_color='crimson', 
        text=[f"+{np.abs(percent_change):.0f}%" if percent_change < 0 
              else f"-{np.abs(percent_change):.0f}%"
              for percent_change in exp_rev_percent_change ],
        textposition='outside', textfont_size=18, textfont_color='red')
])

# Change the bar mode
fig.update_layout(barmode='group', autosize=True, width=800, height=500)

# ================ this is for notebook =========================================================
# fig.show()
# ================ this will display in st.lit ==================================================
st.plotly_chart(fig, use_container_width=True)




# =====================================================================================================================================# # ======================================================================================================================================
# ======================================================================================================================================
# ================================================ BIG FUNCTION ========================================================================
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================
month_string = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
# lean_exp['Months']= ' '
# ls_month
# for i in range(len(lean_exp)):
#     for j,mo in enumerate(ls_month):
        
#         if j ==lean_exp.query('')
#         lean_exp['Months'] = 
        
        
# for y in years:
#     for m,mo in enumerate(ls_month):
#         if lean_exp.query(f'(month =={m+1})&(year=={y})')==mo:
#             lean_exp.query(f'(month =={m+1})&(year=={y})')
            
msg =st.markdown('<h3 class="You have exceeded your monthly income. Balance might be critical !!</h3>', unsafe_allow_html=True)

# This function gets a dataframe as an input and calculates the revenue/expenses rate.
# If the rate exp/salary <80% then all good. if >80 and 100% then we got a problem !
def get_status(df,mon,ye):
    df['rate'] = monthly_expenses_per_cat.rate
    
    cat_list = df.category.unique().tolist()
    max_rate = []
    categ = []
    basic_list = ['food_drink','rent_ai']
    message='You are doing great : expeses/income rate is healthy.'
    #             if((df.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()<0.25) & (df.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()>0.01)):
    #                 print('Well done ! Your revenue for this month is above 75% of your income')
    #             elif ((df.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()<0.6) & (df.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()>0.25)):
    #                 print('You are doing fine. Your revenue for this month is 50% or higher of your income')
    if ((monthly_expenses_per_cat.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()<0.8) & (monthly_expenses_per_cat.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()>0.6)):
        message = ("WARNING ! Your monthly expenses are reaching your monthly income. Check where and if you have spent more than usual or unless your income has drooped.")

    elif ((monthly_expenses_per_cat.query(f'(year=={ye})&(month=={mon})&(Amount<0)').rate.sum()>1)):

        message = ('\n======================================== CRITICAL WARNING ==========================================\n'
                '                  You have exceeded your monthly income. Balance might be critical\n'
                '==================================================================================================\n')
#         message += (f'''You have exceeded your monthly income. Balance might be critical
# ''')
#         message += (f'''====================================================================================================
# ''')

        temp1 = round(monthly_expenses_per_cat.query(f'(year=={ye})&(month=={mon})').rate.sum(),2)
        # here the months with higher expeses are identified
        message += f'''
You have spent {round(100*temp1,2)}%, of your monthly income on :\n
month:  {mon}, 
\nyear:   {ye}
\n categories :\n'''
        for item in cat_list:
#             if item in basic_list:
#                 continue
#             else:
            temp2 = monthly_expenses_per_cat.query(f'(year=={ye})&(month=={mon})&(category=="{item}")').rate.sum()
            max_rate.append(temp2)
            max_value = max(max_rate)
            categ.append(f'{item}')
            max_index = max_rate.index(max_value)              
            message+= f'''
    {item} : {100*temp2}%
'''
# instead of {item} replace line 139 with: {categ[max_index]} : {100*max_value}%
# also the message in line 136 out of the for loop to display only one category and not all.
    return message
# =====================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================
# ================================================ END FUNCTION ========================================================================
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================

box_year = st.selectbox(
    'Choose the year.',
     ls_year)

box_month= st.selectbox(
    'Choose the month.',
     ls_month)

st.write(get_status(df_selection,box_month,box_year))

lean_exp_abs = lean_exp.copy()
lean_exp_abs.Amount = lean_exp_abs.Amount.abs()

# ====================================================================================================================================================
# ====================================================================================================================================================
# ========================================== Line Charts =============================================================================================
# ====================================================================================================================================================
# ====================================================================================================================================================
# st.line_chart(df)

# ls_years_df =[]
# # Splitting the df in years

# for i,year in enumerate(ls_year):
#     # we give the list the name "plot" + the respective year, so the item ls_years_df[0] is the  variable/dataframe named "plot_2019"
#     ls_years_df.append(f'plot_{year}')
#     ls_years_df[i]=df_needed_abs.query(f'year=={year}')


# # for i,year in enumerate(ls_year):
# for df,year in zip(ls_years_df,ls_year):
# #     plt.subplots(figsize=(8, 5))
# #     # since there are only a few month entries from year 2019 , that is how it will look like the 1st graph
# #     plt.plot('month', 'avg_salary', data=df, marker='', color='blue', linewidth=2)
# #     plt.plot( 'month', 'monthly_expenses', data=df, marker='', color='red', linewidth=2)
# #     plt.title(f'Average salary (blue line) with monthly expenses (red line) for the year {year}', fontdict=None, loc='center')
# #     plt.legend()
# #     plt.show()
# #     st.dataframe(df)
# #       st.write(tmp_df)
#     chart_data = df[['avg_salary','monthly_expenses','month']]
#     chart_data = chart_data.rename(columns={'month':'index'}).set_index('index')
#     st.line_chart(chart_data)
# #     st.line_chart(df)
# #     print("\n\n")
    
# ====================================================================================================================================================
# ====================================================================================================================================================
# ================================================= Line Charts ======================================================================================
# ====================================================================================================================================================
# ====================================================================================================================================================


#...To be continued...#

df_selection = lean_exp.query(
    "category == @Category & year == @Year & month ==@Month"
)

st.title("💶 Amount spent in €")
st.markdown("###")

# TOP KPI's

total_expenditures = df_selection_exp['Amount'].sum()
# total_revenues = round(df_selection_rev['avg_salary'].mean()*28,2)
average_value_by_transaction = round(df_selection_exp['Amount'].mean(),2)
# st.write(average_value_by_transaction)

# st.dataframe(df_selection_exp)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Expenditures: "f"{round(total_expenditures,2)} ")
# with middle_column:
#     st.subheader("Total Revenues: "f"{round(avg_revenue,2)} ")
with right_column:
    st.subheader("Avg Transaction Value: "f"{round(average_value_by_transaction,2)} ")
    st.markdown("""---""")

# EXPENDITURES BY CATEGORY [BAR CHART]

exp_by_category = (
    df_selection_exp.groupby(by=["category"]).sum()[["Amount"]].sort_values(by="Amount")
)
fig_category_value = px.bar(
    exp_by_category,
    x="Amount",
    y=exp_by_category.index,
    orientation="h",
    title="<b>Expenditures by Category</b>",
    color_discrete_sequence=["crimson"] * len(exp_by_category),
    template="plotly_white",
)
fig_category_value.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

exp_by_month = (
    df_selection_exp.groupby(by=["month"]).sum()[["Amount"]].sort_values(by="Amount")
)
fig_month_value = px.bar(
    exp_by_month,
    x="Amount",
    y=exp_by_month.index,
    orientation="h",
    title="<b>Expenditures by Month </b>",
    color_discrete_sequence=["crimson"] * len(exp_by_month),
    template="plotly_white",
)
fig_month_value.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_category_value, use_container_width=True)
right_column.plotly_chart(fig_month_value, use_container_width=True)

# if st.checkbox('Monthly status category oriented'):
#     st.dataframe(exp_by_month)


st.markdown("""---""")



# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
