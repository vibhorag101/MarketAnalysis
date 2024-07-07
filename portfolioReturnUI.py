import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import locale

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

locale.setlocale(locale.LC_MONETARY, 'en_IN')

# def create_pie_chart(schemes):
#     labels = list(schemes.keys())
#     values = list(schemes.values())
    
#     fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
#     fig.update_layout(title_text="Scheme Weightages")
#     return fig


def get_nav_data(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['nav'] = df['nav'].astype(float)
    df = df.sort_values('date')
    inception_date = df['date'].min()
    return df, inception_date

def calculate_sip_returns(nav_data, sip_amount, upfront_amount, stepup, start_date, end_date, SIP_Date):
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

    total_investment = upfront_amount
    current_sip_amount = sip_amount

    # do calculation for upfront investment
    units_bought = upfront_amount / nav_data_filtered.iloc[0]['nav']
    units_accumulated = units_bought
    previous_year = start_date.year

    for _, row in last_dates.iloc[:-1].iterrows():
        # Check if a year has passed and increase SIP amount accordingly
        if row['date'].year > previous_year:
            current_sip_amount += current_sip_amount * (stepup / 100)
            previous_year = row['date'].year

        units_bought = current_sip_amount / row['nav']
        units_accumulated += units_bought
        total_investment += current_sip_amount

    final_value = units_accumulated * last_dates.iloc[-1]['nav']
    total_return = (final_value - total_investment) / total_investment * 100

    return total_return, final_value, total_investment


def calculate_portfolio_returns(schemes, sip_amount, upfront_amount, stepup, start_date, end_date, SIP_date, schemes_df):
    scheme_returns = []
    total_investment = 0
    final_value = 0
    inception_dates = []

    for scheme_name, scheme_weight in schemes.items():
        scheme_code = schemes_df[schemes_df['schemeName'] == scheme_name]['schemeCode'].values[0]
        nav_data, inception_date = get_nav_data(scheme_code)
        inception_dates.append((scheme_name, inception_date))
        scheme_return, scheme_final_value, scheme_total_investment = calculate_sip_returns(nav_data, sip_amount * scheme_weight / 100, upfront_amount * scheme_weight / 100, stepup, start_date, end_date, SIP_date)
        scheme_returns.append((scheme_name, scheme_return,scheme_final_value,scheme_total_investment))
        final_value += scheme_final_value
        total_investment += scheme_total_investment

    portfolio_return = (final_value - total_investment) / total_investment * 100
    return portfolio_return, final_value, total_investment, scheme_returns, inception_dates

def update_sip_calculator(*args):
    period = args[0]
    custom_start_date = args[1]
    custom_end_date = args[2]
    SIP_Date = args[3]
    sip_amount = args[4]
    upfront_amount = args[5]
    stepup = args[6]
    schemes_df = args[7]
    schemes = {}
    
    for i in range(8, len(args) - 1, 2):  # Adjust range to account for use_inception_date
        if args[i] and args[i+1]:
            schemes[args[i]] = float(args[i+1])

    use_inception_date = args[-1]  # Get use_inception_date from the last argument

    if not schemes:
        return "Please add at least one scheme.", None, None, None

    total_weight = sum(schemes.values())

    end_date = datetime.now().date()

    if use_inception_date:
        start_date = datetime.strptime(custom_start_date, "%Y-%m-%d").date()
    elif period == "Custom":
        if not custom_start_date or not custom_end_date:
            return "Please provide both start and end dates for custom period.", None, None, None
        start_date = datetime.strptime(custom_start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(custom_end_date, "%Y-%m-%d").date()
    elif period == "YTD":
        start_date = datetime(end_date.year, 1, 1).date()
    elif not period:
        return "Please select a period, provide custom dates, or use the inception date.", None, None, None
    else:
        period_parts = period.split()
        if len(period_parts) < 2:
            return "Invalid period selected.", None, None, None
        
        if 'year' in period_parts[1]:
            years = int(period_parts[0])
            start_date = end_date - timedelta(days=years*365)
        else:
            months = int(period_parts[0])
            start_date = end_date - timedelta(days=months*30)

    try:
        portfolio_return, final_value, total_investment, scheme_returns, inception_dates = calculate_portfolio_returns(schemes, sip_amount, upfront_amount,stepup, start_date, end_date, SIP_Date, schemes_df)
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

    # Check if start_date is before any scheme's inception date
    inception_warnings = []
    earliest_inception_date = max(inception_date for _, inception_date in inception_dates)
    for scheme_name, inception_date in inception_dates:
        if start_date < inception_date.date():
            inception_warnings.append(f"Warning: {scheme_name} inception date ({inception_date.date()}) is after the chosen start date ({start_date}).")

    result = ""
    if inception_warnings:
        result += "The following warnings were found:\n"
        result += "\n".join(inception_warnings) + "\n\n"
        result += f"Possible start date for all chosen schemes is: {earliest_inception_date.date()}\n\n"

    result += f"Portfolio Absolute return: {portfolio_return:.2f}%\n"
    result += f"Total investment: {locale.currency(total_investment,grouping=True)}\n"
    result += f"Final value: {locale.currency(final_value,grouping=True)}\n\n"
    result += "Individual scheme returns:\n"
    for scheme_name, scheme_return, scheme_final_value, scheme_total_investment in scheme_returns:
        result += f"----  {scheme_name}  ----:\n"
        result += f"Return: {scheme_return:.2f}%\n"
        result += f"Total investment: {locale.currency(scheme_total_investment,grouping=True)}\n"
        result += f"Final value: {locale.currency(scheme_final_value,grouping=True)}\n\n"
    # pie_chart = create_pie_chart(schemes)
    # return result, pie_chart, final_value, total_investment
    return result

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
    df["Actions"] = "❌"
    
    # Calculate the sum of weights
    total_weight = df["Weight (%)"].sum()
    
    # Add a row for the total
    total_row = pd.DataFrame({
        "Scheme Name": ["Total"],
        "Weight (%)": [total_weight],
        "Actions": [""]
    })
    
    # Concatenate the original dataframe with the total row
    df = pd.concat([df, total_row], ignore_index=True)
    
    # Add a warning if total weight exceeds 100%
    if total_weight > 100:
        df.loc[df.index[-1], "Actions"] = "⚠️ Exceeds 100%"
    
    return df

def add_scheme_to_list(schemes_list, scheme_name, weight):
    if scheme_name and weight:
        new_list = schemes_list + [(scheme_name, float(weight))]
        return new_list, update_schemes_table(new_list), None, 0
    return schemes_list, update_schemes_table(schemes_list), scheme_name, weight

def update_schemes(schemes_list, updated_data):
    try:
        new_schemes_list = []
        for _, row in updated_data.iterrows():
            scheme_name = row.get('Scheme Name')
            weight = row.get('Weight (%)')
            if scheme_name != 'Total' and weight is not None:
                try:
                    weight_float = float(weight)
                    new_schemes_list.append((scheme_name, weight_float))
                except ValueError:
                    continue
        if not new_schemes_list:
            return schemes_list, update_schemes_table(schemes_list), "No valid schemes found in the table."
        return new_schemes_list, update_schemes_table(new_schemes_list), None
    except Exception as e:
        error_msg = f"Error updating schemes: {str(e)}"
        return schemes_list, update_schemes_table(schemes_list), error_msg

def prepare_inputs(period, custom_start, custom_end, SIP_Date, sip_amount, schemes_list, schemes_df):
    inputs = [period, custom_start, custom_end, SIP_Date, sip_amount, schemes_df]
    for name, weight in schemes_list:
        inputs.extend([name, weight])
    return inputs

def handle_row_selection(schemes_list, evt: gr.SelectData, table_data):
    if evt.index is not None and len(evt.index) > 1:
        column_index = evt.index[1]
        if column_index == 2:  # "Actions" column
            row_index = evt.index[0]
            if row_index < len(table_data) - 1:  # Ensure we're not trying to delete the total row
                # Remove the row
                table_data = table_data.drop(row_index).reset_index(drop=True)
                # Update the schemes_list
                updated_schemes_list = [(row['Scheme Name'], row['Weight (%)']) for _, row in table_data.iterrows() if row['Scheme Name'] != 'Total']
                # Recalculate the total
                return update_schemes_table(updated_schemes_list), updated_schemes_list
    return table_data, schemes_list

def create_ui():
    schemes_df = fetch_scheme_data()

    with gr.Blocks(js=js_func) as app:
        gr.Markdown("# Mutual Fund SIP Returns Calculator")

        with gr.Row():
            period = gr.Dropdown(choices=["YTD", "1 month","3 months","6 months","1 year", "3 years", "5 years", "7 years", "10 years","15 years","20 years", "Custom"], label="Select Period",value="YTD")
            custom_start_date = gr.Textbox(label="Custom Start Date (YYYY-MM-DD)", visible=False)
            custom_end_date = gr.Textbox(label="Custom End Date (YYYY-MM-DD)", visible=False)
            SIP_Date = gr.Dropdown(label="Monthly SIP Date", choices=["start","middle","end"],value="end")
            with gr.Column():
                use_inception_date = gr.Checkbox(label="Use Earliest Inception Date", value=False)
                inception_date_display = gr.Textbox(label="Earliest Inception Date", interactive=False)

        with gr.Row():
            sip_amount = gr.Number(label="SIP Amount (₹)")
            upfront_amount = gr.Number(label="Upfront Investment (₹)",value=0)
            stepup = gr.Number(label="Stepup %",value=0)

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
        
        result = gr.Textbox(label="Results",)
        # pie_chart = gr.Plot(label="Scheme Weightages")
        # final_value = gr.Number(label="Final Value (₹)", interactive=False)
        # total_investment = gr.Number(label="Total Investment (₹)", interactive=False)

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

        def get_earliest_inception_date(schemes_list, schemes_df):
            inception_dates = []
            for scheme_name, _ in schemes_list:
                scheme_code = schemes_df[schemes_df['schemeName'] == scheme_name]['schemeCode'].values[0]
                _, inception_date = get_nav_data(scheme_code)
                inception_dates.append(inception_date)
            return max(inception_dates).strftime("%Y-%m-%d") if inception_dates else ""

        def update_inception_date(use_inception_date, schemes_list, schemes_df):
            if use_inception_date and schemes_list:
                earliest_inception_date = get_earliest_inception_date(schemes_list, schemes_df)
                return gr.update(value=earliest_inception_date, visible=True)
            else:
                return gr.update(value="", visible=False)

        use_inception_date.change(
            update_inception_date,
            inputs=[use_inception_date, schemes_list, gr.State(schemes_df)],
            outputs=inception_date_display
        )

        def prepare_inputs_with_inception(period, custom_start, custom_end, SIP_Date, sip_amount, upfront_amount,stepup, schemes_list, schemes_df, use_inception_date, inception_date_display):
            inputs = [period, custom_start, custom_end, SIP_Date, sip_amount, upfront_amount, stepup, schemes_df]
            for name, weight in schemes_list:
                inputs.extend([name, weight])
            
            inputs.append(use_inception_date)  # Add use_inception_date to the inputs
            if use_inception_date and inception_date_display:
                inputs[1] = inception_date_display  # Replace custom_start with inception_date_display
            
            return inputs

        calculate_button.click(
            lambda *args: update_sip_calculator(*prepare_inputs_with_inception(*args)),
            inputs=[period, custom_start_date, custom_end_date, SIP_Date, sip_amount,upfront_amount,stepup,schemes_list, gr.State(schemes_df), use_inception_date, inception_date_display],
            outputs=[result]
            # outputs=[result, final_value, total_investment]
            # outputs=[result, pie_chart, final_value, total_investment]
        )

    return app

app = create_ui()
app.launch()