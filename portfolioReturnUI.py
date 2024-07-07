import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def get_nav_data(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = df['nav'].astype(float)
    df = df.sort_values('date')
    return df

def calculate_sip_returns(nav_data, sip_amount, start_date, end_date,SIP_Date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    nav_data_filtered = nav_data[(nav_data['date'] >= start_date) & (nav_data['date'] <= end_date)].copy()
    nav_data_filtered['date'] = pd.to_datetime(nav_data_filtered['date'])
    if SIP_Date == 'start':
        last_dates = nav_data_filtered.groupby([nav_data_filtered['date'].dt.year, nav_data_filtered['date'].dt.month]).head(1)
    elif SIP_Date == 'end':
        last_dates = nav_data_filtered.groupby([nav_data_filtered['date'].dt.year, nav_data_filtered['date'].dt.month]).tail(1)
    else:
        last_dates = nav_data_filtered.groupby([nav_data_filtered['date'].dt.year, nav_data_filtered['date'].dt.month]).apply(lambda x: x.iloc[len(x)//2])

    units_accumulated = 0
    total_investment = 0
    
    for _, row in last_dates.iloc[:-1].iterrows():
        units_bought = sip_amount / row['nav']
        units_accumulated += units_bought
        total_investment += sip_amount
    
    final_value = units_accumulated * last_dates.iloc[-1]['nav']
    total_return = (final_value - total_investment) / total_investment * 100
    
    return total_return, final_value, total_investment

def create_pie_chart(schemes):
    labels = list(schemes.keys())
    values = list(schemes.values())
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title_text="Scheme Weightages")
    return fig

def calculate_portfolio_returns(schemes, sip_amount, start_date, end_date, SIP_date,schemes_df):
    scheme_returns = []
    total_investment = 0
    final_value = 0

    for scheme_name, scheme_weight in schemes.items():
        scheme_code = schemes_df[schemes_df['schemeName'] == scheme_name]['schemeCode'].values[0]
        nav_data = get_nav_data(scheme_code)
        scheme_return, scheme_final_value, scheme_total_investment = calculate_sip_returns(nav_data, sip_amount * scheme_weight / 100, start_date, end_date,SIP_date)
        scheme_returns.append((scheme_name, scheme_return))
        final_value += scheme_final_value
        total_investment += scheme_total_investment

    portfolio_return = (final_value - total_investment) / total_investment * 100
    return portfolio_return, final_value, total_investment, scheme_returns

def update_sip_calculator(*args):
    period = args[0]
    custom_start_date = args[1]
    custom_end_date = args[2]
    SIP_Date = args[3]
    sip_amount = args[4]
    schemes_df = args[5]
    schemes = {}
    
    for i in range(6, len(args), 2):
        if args[i] and args[i+1]:
            schemes[args[i]] = float(args[i+1])

    if not schemes:
        return "Please add at least one scheme.", None, None, None

    total_weight = sum(schemes.values())

    end_date = datetime.now().date()

    if period == "Custom":
        if not custom_start_date or not custom_end_date:
            return "Please provide both start and end dates for custom period.", None, None, None
        start_date = datetime.strptime(custom_start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(custom_end_date, "%Y-%m-%d").date()
    
    elif period == "YTD":
        start_date = datetime(end_date.year, 1, 1)

    else:
        # check if string contaiins year
        if 'year' in period.split()[1]:
            years = int(period.split()[0])
            start_date = end_date - timedelta(days=years*365)
        else:
            months = int(period.split()[0])
            start_date = end_date - timedelta(days=months*30)

    try:
        portfolio_return, final_value, total_investment, scheme_returns = calculate_portfolio_returns(schemes, sip_amount, start_date, end_date, SIP_Date,schemes_df)
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

    result = f"Total portfolio SIP return: {portfolio_return:.2f}%\n"
    result += f"Total investment: ₹{total_investment:.2f}\n"
    result += f"Final value: ₹{final_value:.2f}\n\n"
    result += "Individual scheme returns:\n"
    for scheme_name, scheme_return in scheme_returns:
        result += f"{scheme_name}: {scheme_return:.2f}%\n"

    pie_chart = create_pie_chart(schemes)
    
    return result, pie_chart, final_value, total_investment

def fetch_scheme_data():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    schemes = response.json()
    return pd.DataFrame(schemes)

def quick_search_schemes(query, schemes_df):
    if not query:
        return []
    matching_schemes = schemes_df[schemes_df['schemeName'].str.contains(query, case=False, na=False)]
    return matching_schemes['schemeName'].tolist()[:40]

def update_scheme_dropdown(query, schemes_df, key_up_data: gr.KeyUpData):
    schemes = quick_search_schemes(key_up_data.input_value, schemes_df)
    return gr.update(choices=schemes, visible=True)

def update_schemes_list(schemes_list, updated_data):
    new_schemes_list = []
    for _, row in updated_data.iterrows():
        scheme_name = row.get('Scheme Name')
        weight = row.get('Weight (%)')
        action = row.get('Actions')
        if scheme_name and weight is not None and action != '🗑️':  # Only keep rows that aren't marked for deletion
            try:
                weight_float = float(weight)
                new_schemes_list.append((scheme_name, weight_float))
            except ValueError:
                # If weight is not a valid float, skip this row
                continue
    return new_schemes_list

def update_schemes_table(schemes_list):
    df = pd.DataFrame(schemes_list, columns=["Scheme Name", "Weight (%)"])
    df["Actions"] = "❌"  # Use a different emoji to avoid confusion with the deletion mark
    return df

def add_scheme_to_list(schemes_list, scheme_name, weight):
    if scheme_name and weight:
        new_list = schemes_list + [(scheme_name, float(weight))]
        return new_list, update_schemes_table(new_list), None, 0
    return schemes_list, update_schemes_table(schemes_list), scheme_name, weight

def update_schemes(schemes_list, updated_data):
    try:
        new_schemes_list = update_schemes_list(schemes_list, updated_data)
        if not new_schemes_list:
            return schemes_list, update_schemes_table(schemes_list), "No valid schemes found in the table."
        return new_schemes_list, update_schemes_table(new_schemes_list), None
    except Exception as e:
        error_msg = f"Error updating schemes: {str(e)}"
        return schemes_list, update_schemes_table(schemes_list), error_msg

def prepare_inputs(period, custom_start, custom_end,SIP_Date,sip_amount, schemes_list, schemes_df,):
    inputs = [period, custom_start, custom_end,SIP_Date, sip_amount, schemes_df]
    for name, weight in schemes_list:
        inputs.extend([name, weight])
    return inputs

def handle_row_selection(schemes_list, evt: gr.SelectData, table_data):
    # print(f"Event data: {evt}")
    # print(f"Event index: {evt.index}")
    # print(f"Table data: {table_data}")
    
    if evt.index is not None and len(evt.index) > 1:
        column_index = evt.index[1]
        if column_index == 2:  # "Actions" column
            row_index = evt.index[0]
            # Remove the row instead of marking it
            table_data = table_data.drop(row_index).reset_index(drop=True)
            # Update the schemes_list
            updated_schemes_list = [(row['Scheme Name'], row['Weight (%)']) for _, row in table_data.iterrows()]
            return table_data, updated_schemes_list
    return table_data, schemes_list

def update_schemes_table(schemes_list):
    df = pd.DataFrame(schemes_list, columns=["Scheme Name", "Weight (%)"])
    df["Actions"] = "❌"
    return df

def create_ui():
    schemes_df = fetch_scheme_data()

    with gr.Blocks(js=js_func) as app:
        gr.Markdown("# Mutual Fund SIP Returns Calculator")

        with gr.Row():
            period = gr.Dropdown(choices=["YTD", "1 month","3 months","6 months","1 year", "3 years", "5 years", "7 years", "10 years","15 years","20 years", "Custom"], label="Select Period")
            custom_start_date = gr.Textbox(label="Custom Start Date (YYYY-MM-DD)", visible=False)
            custom_end_date = gr.Textbox(label="Custom End Date (YYYY-MM-DD)", visible=False)
            SIP_Date = gr.Dropdown(label="Monthly SIP Date", choices=["start","middle","end"])

        sip_amount = gr.Number(label="SIP Amount (₹)")

        schemes_list = gr.State([])
        
        with gr.Row():
            scheme_dropdown = gr.Dropdown(label="Select Scheme", choices=[], allow_custom_value=True, interactive=True)
            scheme_weight = gr.Slider(minimum=0, maximum=100, step=1, label="Scheme Weight (%)")
            add_button = gr.Button("Add Scheme")

        schemes_table = gr.Dataframe(
            headers=["Scheme Name", "Weight (%)", "Actions"],
            datatype=["str", "number", "str"],
            col_count=(3, "fixed"),
            label="Added Schemes",
            type="pandas",
            interactive=True
        )

        update_button = gr.Button("Update Schemes")
        error_message = gr.Textbox(label="Error", visible=False)
        
        calculate_button = gr.Button("Calculate Returns")
        
        result = gr.Textbox(label="Results")
        pie_chart = gr.Plot(label="Scheme Weightages")
        final_value = gr.Number(label="Final Value (₹)", interactive=False)
        total_investment = gr.Number(label="Total Investment (₹)", interactive=False)

        def update_custom_date_visibility(period):
            return {custom_start_date: gr.update(visible=period=="Custom"),
                    custom_end_date: gr.update(visible=period=="Custom")}

        period.change(update_custom_date_visibility, inputs=[period], outputs=[custom_start_date, custom_end_date])

        scheme_dropdown.key_up(
            fn=update_scheme_dropdown,
            inputs=[scheme_dropdown, gr.State(schemes_df)],
            outputs=scheme_dropdown,
            queue=False,
            show_progress="hidden"
        )

        add_button.click(add_scheme_to_list, 
                         inputs=[schemes_list, scheme_dropdown, scheme_weight], 
                         outputs=[schemes_list, schemes_table, scheme_dropdown, scheme_weight])

        def update_schemes_and_show_error(schemes_list, updated_data):
            new_schemes_list, updated_table, error = update_schemes(schemes_list, updated_data)
            return (
                new_schemes_list,
                updated_table,
                gr.update(value=error, visible=bool(error))
            )

        update_button.click(
            update_schemes_and_show_error,
            inputs=[schemes_list, schemes_table],
            outputs=[schemes_list, schemes_table, error_message]
        )

        schemes_table.select(
                handle_row_selection,
                inputs=[schemes_list, schemes_table],
                outputs=[schemes_table, schemes_list]
        )
        calculate_button.click(
            lambda *args: update_sip_calculator(*prepare_inputs(*args)),
            inputs=[period, custom_start_date, custom_end_date,SIP_Date,sip_amount, schemes_list, gr.State(schemes_df)],
            outputs=[result, pie_chart, final_value, total_investment]
        )

    return app

app = create_ui()
app.launch()