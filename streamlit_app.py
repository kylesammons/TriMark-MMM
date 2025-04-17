import os
import pandas as pd
import streamlit as st
import tempfile
from lightweight_mmm import lightweight_mmm, optimize_media, plot, preprocessing
from datetime import datetime
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import plotly.tools as tls
import seaborn as sns
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import matplotlib.ticker as ticker

# Set Seaborn style for consistency
sns.set_theme(style="whitegrid", palette="muted")

# Set Streamlit page config
st.set_page_config(page_title="TriMark MMM", page_icon=":male_mage:", layout="wide")

# App title
st.title(":male_mage: TriMark MMM")

# Sidebar for inputs
with st.sidebar:
    st.image("/Users/trimark/Desktop/Waves-Logo_Color.svg", width=200)
    st.title("Select Timeframe")
    st.title("Input Budget")
    number_of_weeks = st.number_input(
        "Time Period (in weeks)",
        value=None,
        placeholder="Enter Time Period...",
        step=1,  # ensures integer input
        format="%d",  # forces display as integer
        help="Input the number of weeks you'd like to optimize for",
    )

    st.title("Input Budget")
    budget_input = st.number_input(
        "Planned Budget",
        value=None,
        placeholder="Enter Budget...",
        help="Input total planned budget for all channels included in your data file for the selected time frame",
    )

    st.title("Load Data")
    data_files = st.file_uploader(
        "Upload your data files", accept_multiple_files=True, type=["csv", "xlsx"]
    )

# Initialize session state
if "add_data_files" not in st.session_state:
    st.session_state["add_data_files"] = []

if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

# Display results in the main area
if data_files:
    for data_file in data_files:
        file_name = data_file.name
        if file_name in st.session_state["add_data_files"]:
            continue
        try:
            if not budget_input:
                st.error("Please enter your budget")
                st.stop()

            temp_file_name = None
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, prefix=file_name) as f:
                f.write(data_file.getvalue())
                temp_file_name = f.name

            if temp_file_name:
                st.markdown(f"### Processing {file_name}")
                if file_name.endswith(".csv"):
                    df = pd.read_csv(temp_file_name)
                elif file_name.endswith(".xlsx"):
                    df = pd.read_excel(temp_file_name)

                if df.empty:
                    st.error("Uploaded file is empty. Please upload a valid file.")
                    st.stop()

                st.session_state["add_data_files"].append(file_name)
                st.session_state["df"] = df

                st.success(f"Added {file_name} to model!")

        except Exception as e:
            st.error(f"Error adding {file_name} to model: {e}")
            st.stop()

# Load external factors
external_factors_path = "/Users/trimark/Desktop/SafeStepMMM_ExternalFactors.csv"
external_factors_path_month = "/Users/trimark/Desktop/2023 2024 Marketing Spend Breakout.xlsx - External Factors.csv"
external_factors_df = pd.read_csv(external_factors_path)

if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]
    
    
    # Create two columns, with the first one being 75% and the second one being 25%
    col1, col2 = st.columns([3, 1])  # 3:1 ratio for 75%/25%

    with col1:
        st.dataframe(df.head(104), height=500)

    with col2:
        with st.form(key="response_var_form"):
            response_var = st.selectbox(
                "Response Variable",
                options=df.columns.tolist()[::-1],
                help="Select the response variable for analysis"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            model_type = st.selectbox(
                "Model Type",
                options=["hill_adstock", "adstock", "carryover"],
                help="Select the type of model for analysis"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            sample_size_input = st.selectbox(
                "Sample Size",
                options=[100, 200, 300, 500, 1000],
                help="Select the number of samples for analysis"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            external_factors_input = st.multiselect(
                "External Factors",
                options=[col for col in external_factors_df.columns if col not in['Week_Num', 'Year']],
                help="Choose external factors to include in your model."
            )
            st.markdown("<br>", unsafe_allow_html=True)

            submit_button = st.form_submit_button(label="Run Model")

    if submit_button:
        st.session_state["response_variable"] = response_var

        try:
            progress = st.progress(0)
            progress_text = st.empty()

            # Merge selected external factors
            if external_factors_input:
                selected_factors = external_factors_df[external_factors_input]
                df = pd.merge(df, selected_factors, left_index=True, right_index=True, how="left")

            # Preprocessing data
            progress_text.text("Step 1: Preprocessing data 🙌 ...")
            time.sleep(2)

            # Get a list of all columns excluding 'response_var', 'Year', 'Week', and external factors
            channels = [col for col in df.columns if col not in ['Gross Sales','Net Revenue', 'Week_Num','Week','Year'] + external_factors_input]

            # Prepare media data
            media_data = df[channels].to_numpy()  # Numeric data for media channels
            media_data = media_data.astype(float) # retype to float
            target = df[response_var].to_numpy()  # Response variable data
            target = target.astype(float) # retype to float
            costs = df[channels].sum().to_numpy()  # Cost data for media channels
            extra_features = df[external_factors_input].to_numpy() # Prepare extra features (external factors)
            media_data.shape
            data_size = media_data.shape[0]

            progress.progress(20)
            progress_text.text("Step 2: Scaling data 🫶...")
            time.sleep(2)

            # Split and scale data
            split_point = data_size - number_of_weeks
            media_data_train = media_data[:split_point, ...]
            media_data_test = media_data[split_point:, ...]
            target_train = target[:split_point].reshape(-1)
            extra_features_train = extra_features[:split_point, ...]  # Training data for extra features

            # Scale/Preprocess Data
            media_scaler = preprocessing.CustomScaler(divide_operation=.mean)
            target_scaler = preprocessing.CustomScaler(divide_operation=.mean)
            cost_scaler = preprocessing.CustomScaler(divide_operation=.mean)
            extra_scaler = preprocessing.CustomScaler(divide_operation=.mean)

            media_data_train = media_scaler.fit_transform(media_data_train)
            target_train = target_scaler.fit_transform(target_train)
            costs2 = cost_scaler.fit_transform(costs)
            extra_features_train = extra_scaler.fit_transform(extra_features_train)
            # Check for Nans
            print("NaNs in media_data_train:", np.isnan(media_data_train).any())
            print("Infs in media_data_train:", np.isinf(media_data_train).any())
            print("Negative values in media_data_train:", (media_data_train < 0).sum())

            print("NaNs in extra_features_train:", np.isnan(extra_features_train).any())
            print("Infs in extra_features_train:", np.isinf(extra_features_train).any())
            print("Negative values in extra_features_train:", (extra_features_train < 0).sum())

            print("NaNs in target_train:", np.isnan(target_train).any())
            print("Infs in target_train:", np.isinf(target_train).any())
            print("Negative values in target_train:", (target_train < 0).sum())

            print("NaNs in costs2:", np.isnan(costs2).any())
            print("Infs in costs2:", np.isinf(costs2).any())
            print("Negative values in costs2:", (costs2 < 0).sum())

            # Ensures no nans in media data (avoids Error running MMM: Normal distribution got invalid loc parameter.)
            media_data_train = np.nan_to_num(media_data_train, nan=0.0)
            extra_features_train = np.nan_to_num(extra_features_train, nan=0.0)

            # Initialize the MMM model
            mmm = lightweight_mmm.LightweightMMM(model_name=model_type)

            progress.progress(60)
            progress_text.text("Step 3: Fitting model 🤞...")
            mmm.fit(
                media=media_data_train,
                media_prior=costs2,
                target=target_train,
                extra_features=extra_features_train,  # Include extra features here
                number_warmup=sample_size_input,
                number_samples=sample_size_input,
                number_chains=2,
            )

            # Predictions and insights
            progress.progress(80)
            progress_text.text("Step 4: Training model 💪...")
            time.sleep(2)
            media_contribution, roi_hat = mmm.get_posterior_metrics(
                target_scaler=target_scaler, cost_scaler=cost_scaler
            )

            # Optimization process
            prices = jnp.ones(mmm.n_media_channels)
            n_time_periods = number_of_weeks
            budget = jnp.sum(jnp.dot(prices, media_data.mean(axis=0))) * n_time_periods

            # Run optimization
            solution, kpi_without_optim, previous_budget_allocation = optimize_media.find_optimal_budgets(
                n_time_periods=n_time_periods,
                media_mix_model=mmm,
                budget=budget,
                prices=prices,
                media_scaler=media_scaler,
                target_scaler=target_scaler,
                bounds_lower_pct = 0.15,
                bounds_upper_pct= 0.20,
            )

            # Calculate optimal budget allocation
            optimal_budget_allocation = prices * solution.x

            # Organize results in tabs
            tab1, tab2, tab5, tab3, tab4, tab6 = st.tabs(["Model Fit", "Media Metrics", "Historical", "Attribution", "Optimization", "Prediction"])

            import matplotlib.pyplot as plt


            def apply_custom_plot_formatting(fig, font_size=8, font_color="#30333F", grid_color="#FFFFFF", 
                                            line_width=0.8, grid_line_width=0.5, dpi=300):
                """Apply custom styling to a matplotlib plot."""
                # Set font family and size globally
                plt.rcParams['font.family'] = 'sans-serif'

                # Get all axes from the figure
                axes = fig.get_axes()

                # Apply custom styles to each axis
                for ax in axes:
                    # Customize axis labels and ticks
                    ax.set_ylabel(ax.get_ylabel(), fontsize=font_size, color=font_color)
                    ax.set_xlabel(ax.get_xlabel(), fontsize=font_size, color=font_color)
                    ax.tick_params(axis="x", labelsize=font_size, colors=font_color)
                    ax.tick_params(axis="y", labelsize=font_size, colors=font_color)

                    # Set x-axis ticks starting at 0 and spaced every 4 units
                    x_min, x_max = ax.get_xlim()
                    xticks = np.arange(1, np.ceil(x_max), 4)
                    ax.set_xticks(xticks)

                    # Customize gridlines
                    ax.set_xticks(xticks)  # Ensure ticks are set
                    ax.grid(which='major', axis='x', linestyle='-', alpha=0.1, color=font_color, linewidth=grid_line_width)
                    ax.set_xticks(xticks)  # Again, to ensure the correct ticks are preserved

                    ax.grid(which='major', axis='y', linestyle='-', alpha=0.1, color=font_color, linewidth=grid_line_width)

                    # Set custom spine color
                    for spine in ax.spines.values():
                        spine.set_edgecolor(grid_color)

                    # Adjust title font size if it exists
                    if ax.get_title():
                        ax.title.set_fontsize(font_size)
                        ax.title.set_color(font_color)

                    # Adjust line thickness for all lines in the axes
                    for line in ax.get_lines():
                        line.set_linewidth(line_width)

                    # Customize the legend
                    legend = ax.get_legend()
                    if legend:
                        legend.get_title().set_fontsize(font_size)
                        legend.get_title().set_color(font_color)
                        
                        for text in legend.get_texts():
                            text.set_fontsize(font_size)
                            text.set_color(font_color)

                        for legend_line in legend.get_lines():
                            legend_line.set_linewidth(line_width)

                return fig

            
            def plot_bars_media_metrics(metric, metric_name, channel_names):
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Assign unique colors per channel
                unique_channels = list(set(channel_names))
                colors = plt.cm.get_cmap("tab10", len(unique_channels))  # Use a categorical color map
                color_map = {channel: colors(i) for i, channel in enumerate(unique_channels)}

                # Plot bars
                bars = ax.bar(channel_names, metric, color=[color_map[ch] for ch in channel_names])

                # Add labels
                ax.set_xlabel("Channels")
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name)

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha="right")

                # Add legend
                legend_labels = [plt.Line2D([0], [0], color=color_map[ch], lw=4) for ch in unique_channels]
                ax.legend(legend_labels, unique_channels, title="Channels", loc="upper right")

                return fig
                
            # Generate a color palette based on the number of channels
            channel_colors = sns.color_palette("Set2", len(channels))  # You can change "Set2" to any seaborn palette
            # Convert the color palette to a dictionary mapping channel names to colors
            color_map = {channel: color for channel, color in zip(channels, channel_colors)}

            with tab1:
                with st.spinner("✋ Wait for it..."):
                    st.markdown("### Model Fit Plot")
                    st.info("   📈   Evaluates how well the media mix model aligns with historical performance data.")
                    
                    # Generate the model fit plot
                    fig = plot.plot_model_fit(mmm, target_scaler=target_scaler)
                    
                    # Extract the title string from the plot
                    title = fig.get_axes()[0].get_title()
                    
                    # Clear the title from the plot for cleaner display
                    for ax in fig.get_axes():
                        ax.set_title("")  # Clear the title
                        ax.set_xlabel("Weeks")
                        ax.set_ylabel(response_var)
                        ax.set_ylabel(ax.get_ylabel(), fontsize=4)
                        ax.set_xlabel(ax.get_xlabel(), fontsize=4)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                    
                    # Attempt to extract R2 and MAPE from the title
                    import re
                    r2_value = None
                    mape_value = None
                    match = re.search(r"R2\s*=\s*([\d.]+)\s+MAPE\s*=\s*([\d.]+)%", title, re.IGNORECASE)
                    if match:
                        r2_value = float(match.group(1))
                        mape_value = float(match.group(2))
                    else:
                        st.error("Metrics not found in the plot title. Ensure the plot includes R2 and MAPE.")

                    r2_value_formatted = f"{r2_value * 100:.2f}%"
                    mape_value_formatted = f"{mape_value:.2f}%"

                    # Show the scorecards for R2 and MAPE
                    if r2_value is not None and mape_value is not None:
                        # Style and display R2 and MAPE metrics using the `example` function
                        def scorecards(r2_value, mape_value):
                            col1, col2  = st.columns(2)

                            # Calculate the deltas
                            r2_delta = (r2_value * 100) - 70
                            mape_delta = mape_value - 30
                            
                        
                            # Display the R2 and MAPE metrics
                            col1.metric(
                                label="R2",
                                value=f"{r2_value * 100:.0f}%",
                                delta=f"{r2_delta:+.0f}% v Target (70%)",
                                help= "'Sum of squared Residuals' - Measures how well the model explains the variance in the response variable"
                            )
                            col2.metric(
                                label="MAPE",
                                value=f"{mape_value:.2f}%",
                                delta=f"{mape_delta:+.0f}% v Target (30%)",
                                delta_color="inverse",
                                help= "'Mean Absolute Percentage Error' - Represents the average percentage difference between the predicted and actual values"
                            )
                            
                            # Apply custom styling to metric cards
                            style_metric_cards()
                        scorecards(r2_value, mape_value)
                    with st.expander(label="See explanation", icon="💡"):
                        col1, col2 = st.columns(2)

                        # R² Insights
                        with col1:
                            st.markdown("#### 📊 R2 Insights")
                            st.markdown(f"**R²: {r2_value_formatted}** - Explained Variance")
                            # Determine R² status and row to highlight
                            if r2_value >= 0.90:
                                r2_status = "🌟 Near Perfect Fit"
                                highlight_row_r2 = 0
                            elif 0.80 <= r2_value < 0.90:
                                r2_status = "✅ Highly Accurate Fit"
                                highlight_row_r2 = 1
                            elif 0.70 <= r2_value < 0.80:
                                r2_status = "🟡 Good to Expected Fit"
                                highlight_row_r2 = 2
                            else:
                                r2_status = "❌ Weak Fit"
                                highlight_row_r2 = 3
                            st.progress((r2_value * 100) / 100, text=f"{r2_status}")
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.write("🎯 **Target > 70%** - A higher R² indicates better model accuracy.")
                            
                            # R² table with background color for each range
                            r2_rows = [
                                ["0.90 - 1.00", "Near perfect fit. Explains most variance. Watch for overfitting."],
                                ["0.80 - 0.90", "Highly Accurate fit. Suitable for practical applications."],
                                ["0.70 - 0.80", "Good to expected fit. Validate with industry knowledge."],
                                ["< 0.70", "Weak fit. Needs refinement but may still provide insights."]
                            ]

                            r2_colors = ["#E9F3FC", "#E9F3FC", "#FFFCEC", "#FEEEEE"]  # Green, Green, Yellow, Red

                            r2_table_html = "<table style='width:90%; border-collapse:collapse;'>"
                            r2_table_html += "<tr><th style='text-align:left; padding:5px;font-weight:normal;'>R² Range</th><th style='text-align:left; padding:5px;font-weight:normal';>Interpretation</th></tr>"
                            for i, row in enumerate(r2_rows):
                                background = f"background-color:{r2_colors[i]};" if i == highlight_row_r2 else f"background-color:transparent;"
                                r2_table_html += f"<tr style='{background}'><td style='padding:5px;'>{row[0]}</td><td style='padding:5px;'>{row[1]}</td></tr>"
                            r2_table_html += "</table>"

                            st.markdown(r2_table_html, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)

                        # MAPE Insights
                        with col2:
                            st.markdown("#### 🧮 MAPE Insights")
                            st.markdown(f"**MAPE: {mape_value_formatted}** - Prediction Error")
                            # Determine MAPE status and row to highlight
                            if mape_value < 10:
                                mape_status = "🌟 Excellent Accuracy"
                                highlight_row_mape = 0
                            elif 10 <= mape_value <= 20:
                                mape_status = "✅ Acceptable Accuracy"
                                highlight_row_mape = 1
                            elif 20 < mape_value <= 29.4:
                                mape_status = "🟡 Higher Errors"
                                highlight_row_mape = 2
                            else:
                                mape_status = "❌ Poor Accuracy"
                                highlight_row_mape = 3
                            st.progress(1 - mape_value / 100, text=f"{mape_status}")
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.write("🎯 **Target < 30%** - A lower MAPE indicates smaller prediction errors.")

                            # MAPE table with background color for each range
                            mape_rows = [
                                ["< 10%", "Excellent predictive accuracy."],
                                ["10% - 20%", "Acceptable accuracy. Typical for validated models."],
                                ["20% - %", "Higher errors but might still provide actionable insights."],
                                ["> 30%", "Poor prediction accuracy. Needs improvement."]
                            ]

                            mape_colors = ["#E9F3FC", "#E9F3FC", "#FFFCEC", "#FEEEEE"]  # Green, Green, Yellow, Red

                            mape_table_html = "<table style='width:90%; border-collapse:collapse;'>"
                            mape_table_html += "<tr><th style='text-align:left; padding:5px; font-weight:normal;'>MAPE Range</th><th style='text-align:left; padding:5px; font-weight:normal;'>Interpretation</th></tr>"
                            for i, row in enumerate(mape_rows):
                                background = f"background-color:{mape_colors[i]};" if i == highlight_row_mape else f"background-color:transparent;"
                                mape_table_html += f"<tr style='{background}'><td style='padding:5px;'>{row[0]}</td><td style='padding:5px;'>{row[1]}</td></tr>"
                            mape_table_html += "</table>"

                            st.markdown(mape_table_html, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                    
                    fig.set_size_inches(8, 3)
                    fig = apply_custom_plot_formatting(fig)
                    st.markdown("#### Actual v Predicted KPI Fit")
                    st.pyplot(fig)


            with tab2:
                with st.spinner("✋ Wait for it..."):
                    st.markdown("#### Media Contribution")
                    st.info("   🎯   Displays key performance indicators for each media channel, tracking their effectiveness.")
                    
                    
                    st.markdown("##### Media Contribution Percentage")

                    fig1 = plot.plot_bars_media_metrics(
                        metric=media_contribution, 
                        metric_name="Media Contribution Percentage", 
                        channel_names=channels
                    )

                    # Convert Matplotlib figure to Plotly
                    fig1_plotly = tls.mpl_to_plotly(fig1)

                    # Add formatted data labels to each bar
                    for trace in fig1_plotly.data:
                        if trace.type == 'bar':
                            # Format as percentage with 1 decimal place
                            trace.text = [f"{val * 100:.1f}%" for val in trace.y]
                            trace.textposition = 'outside'
                            trace.textfont = dict(size=18)
                            trace.marker = dict(color='#EF7B49')  # Set bar color

                    # Update layout: channel name font size and height
                    fig1_plotly.update_layout(
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(channels))),
                            ticktext=channels,
                            tickfont=dict(size=18)  # Set x-axis (channel names) font size
                        ),
                        height=fig1_plotly.layout.height * 1.25 if fig1_plotly.layout.height else 600  # Increase height ~25%
                    )

                    st.plotly_chart(fig1_plotly, use_container_width=True)

                    st.markdown("##### Media ROI")

                    fig2 = plot.plot_bars_media_metrics(
                        metric=roi_hat, 
                        metric_name="ROI hat", 
                        channel_names=channels
                    )
                    fig2.suptitle("")  # Clear the figure-level title

                    # Convert Matplotlib figure to Plotly
                    fig2_plotly = tls.mpl_to_plotly(fig2)

                    # Add formatted data labels and set color
                    for trace in fig2_plotly.data:
                        if trace.type == 'bar':
                            # Format as whole numbers
                            trace.text = [f"{int(val)}" for val in trace.y]
                            trace.textposition = 'outside'
                            trace.textfont = dict(size=18)
                            trace.marker = dict(color='#6DC5DB')  # Set bar color

                    # Update layout: channel name font size and height
                    fig2_plotly.update_layout(
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(channels))),
                            ticktext=channels,
                            tickfont=dict(size=18)
                        ),
                        height=fig2_plotly.layout.height * 1.25 if fig2_plotly.layout.height else 600
                    )

                    st.plotly_chart(fig2_plotly, use_container_width=True)


                    with st.expander(label="See explanation", icon="💡"):
                        col1, col2 = st.columns(2)

                        # Media Contribution Percentage Insights
                        with col1:
                            st.markdown("#### 📊 Media Contribution Percentage Insights")
                            st.markdown("""
                                        <ul>
                                            <li><b>What it shows:</b> The percentage contribution of each media channel towards the overall model's predicted outcome.</li>
                                            <li><b>Why it's important:</b> Channels with higher contributions are considered more effective in driving the target variable.</li>
                                            <li><b>Actionable Insight:</b> Focus on high-contribution channels to optimize media strategy.</li>
                                        </ul>
                                        <div style="border-left: 4px solid #66bb6a; background-color: #f9f9f9; padding: 10px; margin-top: 10px;">
                                            <b>Tip:</b> Cross-check high-performing channels with ROI metrics for a balanced investment strategy.
                                        </div>
                                        """, unsafe_allow_html=True)
                            st.write("")

                        # Media ROI Insights
                        with col2:
                            st.markdown("#### 💰 Media ROI Insights")
                            st.markdown("""
                                        <ul>
                                            <li><b>What it shows:</b> The calculated ROI for each media channel, representing the return on investment.</li>
                                            <li><b>Why it's important:</b> A higher ROI indicates better returns for the given spend.</li>
                                            <li><b>Actionable Insight:</b> Invest more in channels with high ROI to maximize cost-effectiveness.</li>
                                        </ul>
                                        <div style="border-left: 4px solid #f39c12; background-color: #f9f9f9; padding: 10px; margin-top: 10px;">
                                            <b>Note:</b> Balance ROI insights with contribution percentages to avoid underfunding impactful channels.
                                        </div>
                                        """, unsafe_allow_html=True)

            with tab3:
                st.markdown("### Contribution Area Plot")
                st.info("   🏆   Breaks down the contribution of each media channel to overall conversions or sales.")
                
                with st.spinner("✋ Wait for it..."):
                    def custom_plot_media_baseline_contribution_area_plot(
                        media_mix_model,
                        target_scaler=None,
                        channel_names=channels,
                        fig_size=(14, 6),
                        font_size=10,
                        dpi=300,
                        font_color="#30333F",
                        grid_color="#FFFFFF",
                    ):
                        """Plots an area chart to visualize weekly media & baseline contribution."""

                        # Set font family globally to sans-serif
                        plt.rcParams['font.family'] = 'sans-serif'

                        # Create media channels & baseline contribution dataframe
                        contribution_df = plot.create_media_baseline_contribution_df(
                            media_mix_model=media_mix_model,
                            target_scaler=target_scaler,
                            channel_names=channel_names
                        ).clip(0)

                        # Validate contribution columns
                        contribution_columns = [col for col in contribution_df.columns if "contribution" in col]
                        if not contribution_columns:
                            st.error("No valid contribution columns found for plotting!")
                            return None

                        contribution_df_for_plot = contribution_df[contribution_columns[::-1]]
                        contribution_df_for_plot["period"] = range(1, len(contribution_df_for_plot) + 1)

                        # Plot the stacked area chart
                        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
                        contribution_df_for_plot.plot.area(x="period", stacked=True, ax=ax)

                        # Customize the plot
                        ax.set_ylabel("Baseline & Media Channels Attribution", fontsize=font_size, color=font_color)
                        ax.set_xlabel("Period", fontsize=font_size, color=font_color)

                        # Customize tick labels
                        ax.tick_params(axis="x", labelsize=font_size, colors=font_color)
                        ax.tick_params(axis="y", labelsize=font_size, colors=font_color)

                        # Customize gridlines
                        ax.grid(axis='x', linestyle='-', alpha=0.1, color=grid_color)
                        ax.grid(axis='y', linestyle='-', alpha=0.1, color=font_color)

                        # Apply the color to all spines (the plot borders)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(grid_color)

                        # Add legend at the bottom of the plot
                        ax.legend(
                            loc="upper center", 
                            bbox_to_anchor=(0.5, -0.15),  # Adjust the vertical position slightly lower
                            fontsize=font_size,
                            labelcolor=font_color,
                            ncol=5,  # Number of columns, adjust based on number of legend items
                            frameon=False,  # Optional: removes the legend box
                            handlelength=2  # Optional: makes the legend entries a bit longer
                        )

                        plt.close()  # Close to avoid Streamlit duplicate rendering
                        return fig

                    # Generate the plot
                    fig3 = custom_plot_media_baseline_contribution_area_plot(
                        media_mix_model=mmm,
                        target_scaler=target_scaler,
                        channel_names=channels
                    )
                    st.pyplot(fig3)
                    
                    # Explanation inside an expander
                    with st.expander(label="See explanation", icon="💡"):
                        st.markdown("#### Attribution Insights")
                        st.write("This chart helps visualize the overall attribution breakdown, distinguishing between media efforts and baseline factors.")
                        st.write("Baseline contribution refers to factors that drive conversions or sales, but are not directly related to media campaigns (e.g., brand strength, seasonality).")
                        st.write("Media contributions highlight the performance of individual media channels and their role in driving outcomes.")
                        st.write("By analyzing this breakdown, you can assess the effectiveness of each channel and understand how external factors (baseline) also influence results.")

            with tab4:
                st.markdown("### Budget Optimization")
                st.info("   ✨   Identifies the ideal media allocation to maximize return on investment.")
                with st.spinner("✋ Wait for it..."):
                    # Convert allocations to percentages of total budget
                    previous_budget_percent = (previous_budget_allocation / budget) * 100
                    optimal_budget_percent = (optimal_budget_allocation / budget) * 100

                    # Use columns layout: chart on the left, scorecards on the right
                    col_chart, col_metrics = st.columns([3, 1])

                    # Plotly chart in the left column
                    with col_chart:
                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=channels,
                            y=previous_budget_percent,
                            name="Previous Budget Allocation",
                            text=[f"{value:.1f}%" for value in previous_budget_percent],
                            textposition='outside',
                            marker_color='rgb(26, 118, 255)',
                            textfont=dict(size=16)
                        ))

                        fig.add_trace(go.Bar(
                            x=channels,
                            y=optimal_budget_percent,
                            name="Optimal Budget Allocation",
                            text=[f"{value:.1f}%" for value in optimal_budget_percent],
                            textposition='outside',
                            marker_color='rgb(55, 83, 109)',
                            textfont=dict(size=16)
                        ))

                        # Customize layout
                        fig.update_layout(
                            barmode='group',
                            xaxis_title="",
                            yaxis_title="Budget Allocation (%)",
                            yaxis_ticksuffix="%",
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                            height=500,
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=0, b=20, l=20, r=20)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Scorecards in the right column
                    with col_metrics:
                        st.metric(
                            label="Pre-Optimization Target",
                            value=f"{int(kpi_without_optim * -1):,}",
                            help="The target KPI before optimization"
                        )
                        st.metric(
                            label="Post-Optimization Target",
                            value=f"{int(solution['fun'] * -1):,}",
                            delta=f"{((solution['fun'] * -1 - kpi_without_optim * -1) / (kpi_without_optim * -1)) * 100:.1f}% v Pre-Optimization Target",
                            help="The target KPI after optimization"
                        )
                        st.metric(
                            label="Recommended Budget",
                            value=f"${int(budget):,}",
                            delta=f"{((budget - budget_input) / budget)*100:.1f}% v Planned Budget",
                            help="Budget after optimization"
                        )

                    # Table displayed below
                    st.markdown("#### Detailed Budget Allocation Table")
                    difference = optimal_budget_allocation - previous_budget_allocation
                    percent_change = (difference / previous_budget_allocation) * 100
                    budget_data = pd.DataFrame({
                        "Channel": channels,
                        "Previous Budget Allocation ($)": [f"${value:0,.0f}" for value in previous_budget_allocation],
                        "Optimal Budget Allocation ($)": [f"${value:0,.0f}" for value in optimal_budget_allocation],
                        "Difference ($)": [f"${value:0,.0f}" for value in difference],
                        "Percent Change": [f"{value:0,.0f}%" for value in percent_change]
                    })
                    st.dataframe(budget_data, use_container_width=True)

                    # Add an expander for detailed explanations
                    with st.expander(label="See explanation", expanded=False, icon="💡"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🎯 Optimization Insights")
                            st.markdown("""
                            <ul>
                                <li><b>Pre-Optimization:</b> The initial target KPI based on the current budget allocation.</li>
                                <li><b>Post-Optimization:</b> The optimized target KPI after re-allocating the budget across channels.</li>
                                <li><b>Actionable Insight:</b> Use the optimized budget allocation to maximize returns.</li>
                            </ul>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown("#### 💰 Budget Allocation Insights")
                            st.markdown("""
                            <ul>
                                <li><b>Budget Change:</b> The difference between the planned and optimized budgets.</li>
                                <li><b>Why it matters:</b> Optimized allocation ensures that the spending aligns with ROI goals.</li>
                                <li><b>Actionable Insight:</b> Reallocate resources to underperforming channels to maximize effectiveness.</li>
                            </ul>
                            """, unsafe_allow_html=True)

            with tab5:
                st.markdown("### Historical Media Mix")
                st.info("   🧾   Analyzes past media spend and performance to understand trends and outcomes.")
                
                total_response_var = df[response_var].sum()
                response_var_per_period = total_response_var / len(df['Week'])
                total_spend = df[channels].sum(axis=0)  # Sum each channel separately
                total_spend_sum = total_spend.sum()  # Total spend across all channels
                spend_per_period = total_spend_sum / len(df['Week'])

                # Scorecards Section
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label=f"Total {response_var}", value=f"{total_response_var:0,.0f}")
                with col2:
                    st.metric(label=f"{response_var} per Week", value=f"{response_var_per_period:0,.0f}")
                with col3:
                    st.metric(label="Total Spend", value=f"${total_spend_sum:0,.0f}")
                with col4:
                    st.metric(label="Spend per Week", value=f"${spend_per_period:0,.0f}")
                
                # Pie Chart and Table Section in two columns
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Historical Budget Allocation")
                    # Calculate total budget per channel (assuming budget information is included in the dataframe)
                    channel_budget = df[channels].sum()  # Summing the values for each channel
                    
                    # Create a pie chart with the dynamic color map
                    fig_pie = px.pie(
                        names=channel_budget.index,
                        values=channel_budget.values,
                        color=channel_budget.index,
                        color_discrete_map=color_map  # Use the dynamic color map for consistency
                    )
                    # Adjust the layout for height and padding
                    fig_pie.update_layout(
                        height=300,  # Adjust the height as needed
                        margin=dict(t=20, b=20, l=20, r=200)  # Adjust padding (top, bottom, left, right)
                    )
                    st.plotly_chart(fig_pie)
                
                    with col2:
                        st.markdown("### Quarterly Breakdown")

                        # Extract the numeric week from the 'Week' string (e.g., 'w1 2024' → 1)
                        def extract_week_number(week_str):
                            match = re.search(r'w(\d+)', str(week_str).lower())
                            return int(match.group(1)) if match else None

                        df['Week_Number'] = df['Week'].apply(extract_week_number)

                        # Convert to Quarter based on week number
                        def week_to_quarter(week):
                            if 1 <= week <= 13:
                                return 'Q1'
                            elif 14 <= week <= 26:
                                return 'Q2'
                            elif 27 <= week <= 39:
                                return 'Q3'
                            elif 40 <= week <= 53:
                                return 'Q4'
                            else:
                                return 'Unknown'

                        df['Quarter'] = df['Week_Number'].apply(week_to_quarter)

                        # Calculate total spend and response
                        df['total_spend_sum'] = df[channels].sum(axis=1)
                        df['total_response_var'] = df[response_var]

                        # Group by Year and Quarter
                        df_grouped = df.groupby(['Year', 'Quarter']).agg(
                            total_spend_sum=('total_spend_sum', 'sum'),
                            total_response_var=('total_response_var', 'sum')
                        ).reset_index()

                        # Rename for display
                        df_grouped.rename(columns={
                            'total_spend_sum': 'Spend',
                            'total_response_var': response_var
                        }, inplace=True)

                        # Format display numbers
                        df_grouped['Spend'] = df_grouped['Spend'].apply(lambda x: f"${x:0,.0f}")
                        df_grouped[response_var] = df_grouped[response_var].apply(lambda x: f"{x:0,.0f}")

                        # Show table
                        st.dataframe(df_grouped, use_container_width=True)

                        
                # Line Chart Section
                st.markdown("### Media Allocation Over Time")
                with st.spinner("✋ Wait for it..."):
                    # Create the line chart with custom colors using the generated color map
                    historical_data = df[channels]  # Selecting the channel columns
                    st.line_chart(historical_data, use_container_width=True)

            with tab6:
                st.markdown("### Predicted KPI")
                st.info("   🎱   Projects future performance based on historical data and model insights.")
                with st.spinner("✋ Wait for it..."):

                    new_predictions = mmm.predict(media=media_scaler.transform(media_data_test))
                    target_transformed = target_scaler.transform(target[split_point:])
                    # Plot the results
                    fig5 = plot.plot_out_of_sample_model_fit(
                        out_of_sample_predictions=new_predictions,
                        out_of_sample_target=target_transformed
                    )
                    fig5.set_size_inches(8, 3)
                    fig5 = apply_custom_plot_formatting(fig5)
                    st.pyplot(fig5)
            progress.progress(100)
            progress_text.text("Model Complete! 100%")

        except Exception as e:
            st.error(f"Error running MMM: {e}")
            st.stop()
