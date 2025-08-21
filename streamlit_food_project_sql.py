import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set the page configuration for a wide layout
st.set_page_config(
    page_title="Local Food Wastage Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A basic assumption is that the various CSV files exist in the same directory.
try:
    df_merged = pd.read_csv('merged_data.csv')
    df_claims = pd.read_csv('claims_cleaned.csv')
    df_food = pd.read_csv('food_listings_cleaned.csv')
    df_providers = pd.read_csv('providers_data_cleaned.csv')
    df_receivers = pd.read_csv('receivers_data_cleaned.csv')
    data_loaded = True
except FileNotFoundError:
    st.error("Error: One or more of the required CSV files (merged_data.csv, claims_cleaned.csv, food_listings_cleaned.csv, providers_data_cleaned.csv, receivers_data_cleaned.csv) were not found. Please ensure they are in the same directory as this script.")
    data_loaded = False
    
def get_time_of_day(hour):
    """Categorizes the hour into a time of day."""
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Clean and prepare the merged data for analysis
def prepare_data(df):
    """
    Performs data cleaning and preparation steps on the DataFrame.
    """
    # Standardize column names by removing leading/trailing spaces
    df.columns = df.columns.str.strip()
    
    # Convert 'timestamp' to datetime for time-based analysis
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Create a new 'time_of_day' column based on the timestamp hour
        df['time_of_day'] = df['timestamp'].dt.hour.apply(get_time_of_day)
    
    # Fill any NaN values in key columns to prevent errors in filtering
    df[['location', 'provider_name', 'food_type', 'meal_type', 'food_status']] = \
        df[['location', 'provider_name', 'food_type', 'meal_type', 'food_status']].fillna('Unknown')
        
    return df

# Apply data cleaning to the merged dataframe right after loading
if data_loaded:
    df_merged = prepare_data(df_merged)

# This function is no longer cached to avoid the threading issue
def create_and_populate_db():
    """
    Creates an in-memory SQLite database and populates it with data from DataFrames.
    """
    conn = sqlite3.connect(':memory:')
    
    # Write each DataFrame to a separate table
    if data_loaded:
        df_merged.to_sql('merged_data', conn, index=False, if_exists='replace')
        df_claims.to_sql('claims', conn, index=False, if_exists='replace')
        df_food.to_sql('food_listings', conn, index=False, if_exists='replace')
        df_providers.to_sql('providers', conn, index=False, if_exists='replace')
        df_receivers.to_sql('receivers', conn, index=False, if_exists='replace')
        
    return conn

# Connect to the database
if data_loaded:
    conn = create_and_populate_db()

# --- Streamlit UI Components ---

# Main title and introduction
st.title("Local Food Wastage Management System")
st.markdown("""
This application is a user interface for managing and analyzing food donations. 
It helps connect surplus food providers with those in need, reduces food waste, and provides
insights into donation trends.
""")

# Use a radio button in the sidebar to switch between views
view_mode = st.sidebar.radio(
    "Select View",
    ('Analytics Dashboard', 'View Raw Data', 'Python Charts')
)

# --- Analytics Dashboard View ---
if view_mode == 'Analytics Dashboard':
    if data_loaded:
        # --- Sidebar Filters (as per project document) ---
        st.sidebar.header("Filter Donations")

        # Get unique values for each filter from the merged DataFrame for the selectboxes
        cities = sorted(df_merged['location'].unique())
        providers = sorted(df_merged['provider_name'].unique())
        food_types = sorted(df_merged['food_type'].unique())
        meal_types = sorted(df_merged['meal_type'].unique())
        time_of_days = sorted(df_merged['time_of_day'].unique())

        # Create select boxes for user filtering
        selected_city = st.sidebar.selectbox("City", ["All"] + list(cities))
        selected_provider = st.sidebar.selectbox("Provider", ["All"] + list(providers))
        selected_food_type = st.sidebar.selectbox("Food Type", ["All"] + list(food_types))
        selected_meal_type = st.sidebar.selectbox("Meal Type", ["All"] + list(meal_types))
        selected_time_of_day = st.sidebar.selectbox("Time of Day", ["All"] + list(time_of_days))

        # Build the WHERE clause for the SQL query based on selected filters
        where_clause = "WHERE 1=1"
        if selected_city != "All":
            where_clause += f" AND location = '{selected_city}'"
        if selected_provider != "All":
            where_clause += f" AND provider_name = '{selected_provider}'"
        if selected_food_type != "All":
            where_clause += f" AND food_type = '{selected_food_type}'"
        if selected_meal_type != "All":
            where_clause += f" AND meal_type = '{selected_meal_type}'"
        if selected_time_of_day != "All":
            where_clause += f" AND time_of_day = '{selected_time_of_day}'"

        # Query the database for the filtered data
        query_filtered_data = f"SELECT * FROM merged_data {where_clause};"
        filtered_df = pd.read_sql_query(query_filtered_data, conn)

        # --- Key Performance Indicators (KPIs) Section ---
        st.header("Key Performance Indicators (KPIs)")
        
        # Calculate the metrics
        total_listings = filtered_df.shape[0]
        unique_providers = filtered_df['provider_name'].nunique()
        unique_receivers = filtered_df['receiver_name'].nunique()
        
        # Calculate percentages, handling division by zero
        total_claims = len(filtered_df)
        if total_claims > 0:
            completed_claims = (filtered_df['status'] == 'Completed').sum()
            pending_claims = (filtered_df['status'] == 'Pending').sum()
            cancelled_claims = (filtered_df['status'] == 'Cancelled').sum()
            
            success_percentage = (completed_claims / total_claims) * 100
            pending_percentage = (pending_claims / total_claims) * 100
            cancellation_percentage = (cancelled_claims / total_claims) * 100

            expired_food_count = (filtered_df['food_status'] == 'expired').sum()
            expired_food_percentage = (expired_food_count / total_claims) * 100
        else:
            success_percentage = 0
            pending_percentage = 0
            cancellation_percentage = 0
            expired_food_percentage = 0

        # Calculate most active time of day and most listed food type
        most_active_time = filtered_df['time_of_day'].mode().iloc[0] if not filtered_df['time_of_day'].empty else "N/A"
        most_listed_food = filtered_df['food_name'].mode().iloc[0] if not filtered_df['food_name'].empty else "N/A"
        
        # Display the KPIs in columns for a clean layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Listings", f"{total_listings:,}")
        with col2:
            st.metric("Unique Providers", unique_providers)
        with col3:
            st.metric("Unique Receivers", unique_receivers)
        with col4:
            st.metric("Most Listed Food Type", most_listed_food)
            
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Success %", f"{success_percentage:.2f}%")
        with col6:
            st.metric("Pending %", f"{pending_percentage:.2f}%")
        with col7:
            st.metric("Cancellation %", f"{cancellation_percentage:.2f}%")
        with col8:
            st.metric("Expired Food %", f"{expired_food_percentage:.2f}%")
            
        st.metric("Most Active Time of Day", most_active_time)


        # --- Main Content Display ---
        st.header("Filtered Food Listings")
        
        if filtered_df.empty:
            st.warning("No data found for the selected filters.")
        else:
            # Display the filtered data in a user-friendly way
            st.dataframe(filtered_df.drop(columns=['address', 'receiver_contact', 'provider_city', 'receiver_city'], errors='ignore'), use_container_width=True)

            # Display Provider Contact Details for direct coordination
            st.subheader("Provider Contact Details")
            # Show only unique providers with their contact info
            provider_contact_df = filtered_df[['provider_name', 'provider_contact']].drop_duplicates()
            if provider_contact_df.empty:
                st.info("No providers to display contact details for.")
            else:
                st.dataframe(provider_contact_df, use_container_width=True)

        # --- Answering SQL Queries ---

        st.header("Project Insights (SQL Queries)")
        st.markdown("Select a question from the dropdown to see the analysis and the query used to generate it.")

        # Dictionary mapping question text to a unique identifier
        questions = {
            "Q1: Providers and Receivers by City": "q1",
            "Q2: Food Contribution by Provider Type": "q2",
            "Q4: Top Receivers by Claims": "q4",
            "Q5: Total Food Quantity Available": "q5",
            "Q6: Cities with Most Food Listings": "q6",
            "Q8: Food Claims by Item": "q8",
            "Q9: Providers with Highest Successful Claims": "q9",
            "Q10: Claim Status Distribution": "q10",
            "Q11: Average Quantity Claimed per Receiver": "q11",
            "Q12: Most Claimed Meal Type": "q12",
            "Q13: Total Donated Quantity by Provider": "q13",
        }

        # Create a selectbox for the user to choose a question
        selected_question = st.selectbox("Select a Question", list(questions.keys()))
        
        # Initialize query and result_type to None
        query = None
        result_type = None

        # Determine the SQL query based on the selected question
        if selected_question == "Q1: Providers and Receivers by City":
            query = f"""
                SELECT
                    COALESCE(p.location, r.receiver_city) AS City,
                    COUNT(DISTINCT p.provider_id) AS "Number of Providers",
                    COUNT(DISTINCT r.receiver_id) AS "Number of Receivers"
                FROM merged_data AS p
                LEFT JOIN merged_data AS r ON p.location = r.receiver_city
                {where_clause}
                GROUP BY City
                ORDER BY "Number of Providers" DESC, "Number of Receivers" DESC;
            """
            result_type = 'dataframe'

        elif selected_question == "Q2: Food Contribution by Provider Type":
            query = f"""
                SELECT
                    provider_type,
                    SUM(quantity) AS "Total Quantity Donated"
                FROM merged_data
                {where_clause}
                GROUP BY provider_type
                ORDER BY "Total Quantity Donated" DESC;
            """
            result_type = 'dataframe'

        elif selected_question == "Q4: Top Receivers by Claims":
            query = f"""
                SELECT
                    receiver_name AS "Receiver Name",
                    SUM(quantity) AS "Total Quantity Claimed"
                FROM merged_data
                {where_clause}
                GROUP BY receiver_name
                ORDER BY "Total Quantity Claimed" DESC
                LIMIT 10;
            """
            result_type = 'dataframe'

        elif selected_question == "Q5: Total Food Quantity Available":
            query = f"""
                SELECT SUM(quantity) FROM merged_data {where_clause};
            """
            result_type = 'metric'

        elif selected_question == "Q6: Cities with Most Food Listings":
            query = f"""
                SELECT
                    location AS "City",
                    COUNT(*) AS "Number of Listings"
                FROM merged_data
                {where_clause}
                GROUP BY location
                ORDER BY "Number of Listings" DESC
                LIMIT 10;
            """
            result_type = 'dataframe'

        elif selected_question == "Q8: Food Claims by Item":
            query = f"""
                SELECT
                    food_name AS "Food Name",
                    COUNT(claim_id) AS "Number of Claims"
                FROM merged_data
                {where_clause}
                GROUP BY food_name
                ORDER BY "Number of Claims" DESC;
            """
            result_type = 'dataframe'

        elif selected_question == "Q9: Providers with Highest Successful Claims":
            query = f"""
                SELECT
                    provider_name AS "Provider Name",
                    COUNT(claim_id) AS "Number of Completed Claims"
                FROM merged_data
                WHERE status = 'Completed' {where_clause.replace('WHERE 1=1', 'AND 1=1')}
                GROUP BY provider_name
                ORDER BY "Number of Completed Claims" DESC
                LIMIT 10;
            """
            result_type = 'dataframe'

        elif selected_question == "Q10: Claim Status Distribution":
            query = f"""
                SELECT
                    status,
                    CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM merged_data {where_clause}) AS REAL) AS Percentage
                FROM merged_data
                {where_clause}
                GROUP BY status;
            """
            result_type = 'chart'

        elif selected_question == "Q11: Average Quantity Claimed per Receiver":
            query = f"""
                SELECT
                    AVG(quantity_claimed)
                FROM (
                    SELECT
                        receiver_id,
                        SUM(quantity) AS quantity_claimed
                    FROM merged_data
                    {where_clause}
                    GROUP BY receiver_id
                ) as subquery;
            """
            result_type = 'metric'

        elif selected_question == "Q12: Most Claimed Meal Type":
            query = f"""
                SELECT
                    meal_type AS "Meal Type",
                    COUNT(claim_id) AS "Number of Claims"
                FROM merged_data
                {where_clause}
                GROUP BY meal_type
                ORDER BY "Number of Claims" DESC;
            """
            result_type = 'dataframe'

        elif selected_question == "Q13: Total Donated Quantity by Provider":
            query = f"""
                SELECT
                    provider_name AS "Provider Name",
                    SUM(quantity) AS "Total Quantity Donated"
                FROM merged_data
                {where_clause}
                GROUP BY provider_name
                ORDER BY "Total Quantity Donated" DESC;
            """
            result_type = 'dataframe'

        # Display the query and the button to execute it
        if query:
            st.subheader("SQL Query for this analysis:")
            st.code(query, language='sql')
            
            if st.button('Run Query'):
                st.subheader("Query Results:")
                if result_type == 'dataframe':
                    result_df = pd.read_sql_query(query, conn)
                    if result_df.empty:
                        st.info("No results found for this query with the current filters.")
                    else:
                        st.dataframe(result_df, use_container_width=True)
                
                elif result_type == 'chart':
                    result_df = pd.read_sql_query(query, conn)
                    if result_df.empty:
                        st.info("No results found for this query with the current filters.")
                    else:
                        st.bar_chart(result_df.set_index(result_df.columns[0]))
                        
                elif result_type == 'metric':
                    result = pd.read_sql_query(query, conn).iloc[0, 0]
                    if result is not None:
                        # Specific formatting for each metric
                        if selected_question == "Q5: Total Food Quantity Available":
                            st.metric(label="Total Food Quantity", value=f"{result:,} units")
                        elif selected_question == "Q11: Average Quantity Claimed per Receiver":
                            st.metric(label="Average Quantity per Receiver", value=f"{result:.2f} units")
                    else:
                        st.info("No results found for this query with the current filters.")
    else:
        st.stop()

# --- View Raw Data View ---
elif view_mode == 'View Raw Data':
    st.header("Raw Data Tables")
    
    # Use a selectbox to choose which table to display
    table_choice = st.sidebar.selectbox(
        "Select Table to View",
        ('merged_data.csv', 'claims_cleaned.csv', 'food_listings_cleaned.csv', 'providers_data_cleaned.csv', 'receivers_data_cleaned.csv')
    )
    
    st.subheader(f"Data from {table_choice}")
    
    # Display the selected dataframe
    if table_choice == 'merged_data.csv':
        st.dataframe(df_merged, use_container_width=True)
    elif table_choice == 'claims_cleaned.csv':
        st.dataframe(df_claims, use_container_width=True)
    elif table_choice == 'food_listings_cleaned.csv':
        st.dataframe(df_food, use_container_width=True)
    elif table_choice == 'providers_data_cleaned.csv':
        st.dataframe(df_providers, use_container_width=True)
    elif table_choice == 'receivers_data_cleaned.csv':
        st.dataframe(df_receivers, use_container_width=True)
        
# --- Python Charts View ---
elif view_mode == 'Python Charts':
    st.header("Python Charts for Data Analysis")
    st.markdown("Select a chart type to visualize data distributions and trends.")

    # Dictionary mapping chart question to a unique identifier
    chart_questions = {
        "1. Distribution of Food Types": "food_type_dist",
        "2. Distribution of Meal Types": "meal_type_dist",
        "3. Distribution of Provider Types": "provider_type_dist",
        "4. Distribution of Receiver Types": "receiver_type_dist",
        "5. Bivariate analysis: Provider Type vs. Status": "provider_status",
        "6. Bivariate analysis: Food Type vs. Status": "food_status",
        "7. Bivariate analysis: Meal Type vs. Status": "meal_status",
        "8. Bivariate analysis: Provider Type vs. Receiver Type": "provider_receiver",
        "9. Average Quantity of Food Items by Food Name": "avg_quantity_food_name",
        "10. Distribution of Quantity by Provider Type (Boxplot)": "quantity_provider_type",
        "11. Distribution of Quantity by Food Type (Boxplot)": "food_type_boxplot",
        "12. Distribution of Quantity by Meal Type (Boxplot)": "meal_type_boxplot",
        "13. Distribution of Quantity by Claim Status (Boxplot)": "claim_status_boxplot",
        "14. Top 10 Locations": "top_locations",
        "15. Top 10 Receiver Cities": "top_receiver_cities",
        "16. Number of Claims by Day of the Month": "claims_by_day_of_month",
        "17. Claim Status Distribution by Day of Month (Line Chart)": "claim_status_by_day",
        "18. Claim Status Distribution by Time of Day": "claim_status_by_time_of_day",
        "19. Number of Claims by Hour of the Day": "claims_by_hour",
        "20. Claim Status Distribution by Hour of Day (Line Chart)": "claim_status_by_hour"
    }

    selected_chart = st.selectbox("Select a Chart", list(chart_questions.keys()))

    if selected_chart == "1. Distribution of Food Types":
        st.subheader("Distribution of Food Types")
        food_type_counts = df_merged['food_type'].value_counts()
        
        # Create a pie chart using matplotlib with a darker palette
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(food_type_counts, labels=food_type_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
        ax.set_title('Distribution of Food Types')
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="food_type_distribution.png",
            mime="image/png"
        )

    elif selected_chart == "2. Distribution of Meal Types":
        st.subheader("Distribution of Meal Types")
        meal_type_counts = df_merged['meal_type'].value_counts().reset_index()
        meal_type_counts.columns = ['Meal Type', 'Count']
        st.bar_chart(meal_type_counts.set_index('Meal Type'))
        st.download_button(
            label="Download Data as CSV",
            data=meal_type_counts.to_csv(index=False).encode('utf-8'),
            file_name="meal_type_distribution.csv",
            mime="text/csv"
        )

    elif selected_chart == "3. Distribution of Provider Types":
        st.subheader("Distribution of Provider Types")
        provider_type_counts = df_providers['type'].value_counts().reset_index()
        provider_type_counts.columns = ['Provider Type', 'Count']
        st.bar_chart(provider_type_counts.set_index('Provider Type'))
        st.download_button(
            label="Download Data as CSV",
            data=provider_type_counts.to_csv(index=False).encode('utf-8'),
            file_name="provider_type_distribution.csv",
            mime="text/csv"
        )

    elif selected_chart == "4. Distribution of Receiver Types":
        st.subheader("Distribution of Receiver Types")
        receiver_type_counts = df_receivers['type'].value_counts().reset_index()
        receiver_type_counts.columns = ['Receiver Type', 'Count']
        st.bar_chart(receiver_type_counts.set_index('Receiver Type'))
        st.download_button(
            label="Download Data as CSV",
            data=receiver_type_counts.to_csv(index=False).encode('utf-8'),
            file_name="receiver_type_distribution.csv",
            mime="text/csv"
        )

    elif selected_chart == "5. Bivariate analysis: Provider Type vs. Status":
        st.subheader("Bivariate analysis: Provider Type vs. Status (Clustered Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_merged, x='provider_type', hue='status', ax=ax, palette='mako')
        ax.set_title('Provider Type vs. Status')
        ax.set_xlabel('Provider Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="provider_type_status.png",
            mime="image/png"
        )

    elif selected_chart == "6. Bivariate analysis: Food Type vs. Status":
        st.subheader("Bivariate analysis: Food Type vs. Status (Clustered Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_merged, x='food_type', hue='status', ax=ax, palette='mako')
        ax.set_title('Food Type vs. Status')
        ax.set_xlabel('Food Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="food_type_status.png",
            mime="image/png"
        )

    elif selected_chart == "7. Bivariate analysis: Meal Type vs. Status":
        st.subheader("Bivariate analysis: Meal Type vs. Status (Clustered Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_merged, x='meal_type', hue='status', ax=ax, palette='mako')
        ax.set_title('Meal Type vs. Status')
        ax.set_xlabel('Meal Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="meal_type_status.png",
            mime="image/png"
        )

    elif selected_chart == "8. Bivariate analysis: Provider Type vs. Receiver Type":
        st.subheader("Bivariate analysis: Provider Type vs. Receiver Type (Clustered Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_merged, x='provider_type', hue='receiver_type', ax=ax, palette='viridis')
        ax.set_title('Provider Type vs. Receiver Type')
        ax.set_xlabel('Provider Type')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="provider_receiver_type.png",
            mime="image/png"
        )

    elif selected_chart == "9. Average Quantity of Food Items by Food Name":
        st.subheader("Average Quantity of Food Items by Food Name")
        avg_quantity = df_merged.groupby('food_name')['quantity'].mean().sort_values(ascending=False).reset_index()
        avg_quantity.columns = ['Food Name', 'Average Quantity']
        st.bar_chart(avg_quantity.set_index('Food Name'))
        st.download_button(
            label="Download Data as CSV",
            data=avg_quantity.to_csv(index=False).encode('utf-8'),
            file_name="avg_quantity_by_food_name.csv",
            mime="text/csv"
        )

    elif selected_chart == "10. Distribution of Quantity by Provider Type (Boxplot)":
        st.subheader("Distribution of Quantity by Provider Type")
        fig, ax = plt.subplots()
        sns.boxplot(x='provider_type', y='quantity', data=df_merged, ax=ax, palette='viridis')
        plt.title('Distribution of Quantity by Provider Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="quantity_by_provider_type.png",
            mime="image/png"
        )
        
    elif selected_chart == "11. Distribution of Quantity by Food Type (Boxplot)":
        st.subheader("Distribution of Quantity by Food Type")
        fig, ax = plt.subplots()
        sns.boxplot(x='food_type', y='quantity', data=df_merged, ax=ax, palette='viridis')
        plt.title('Distribution of Quantity by Food Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="quantity_by_food_type.png",
            mime="image/png"
        )
        
    elif selected_chart == "12. Distribution of Quantity by Meal Type (Boxplot)":
        st.subheader("Distribution of Quantity by Meal Type")
        fig, ax = plt.subplots()
        sns.boxplot(x='meal_type', y='quantity', data=df_merged, ax=ax, palette='viridis')
        plt.title('Distribution of Quantity by Meal Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="quantity_by_meal_type.png",
            mime="image/png"
        )
        
    elif selected_chart == "13. Distribution of Quantity by Claim Status (Boxplot)":
        st.subheader("Distribution of Quantity by Claim Status")
        fig, ax = plt.subplots()
        sns.boxplot(x='status', y='quantity', data=df_merged, ax=ax, palette='viridis')
        plt.title('Distribution of Quantity by Claim Status')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="quantity_by_claim_status.png",
            mime="image/png"
        )
        
    elif selected_chart == "14. Top 10 Locations":
        st.subheader("Top 10 Locations")
        top_locations = df_merged['location'].value_counts().nlargest(10).reset_index()
        top_locations.columns = ['City', 'Number of Listings']
        st.bar_chart(top_locations.set_index('City'))
        st.download_button(
            label="Download Data as CSV",
            data=top_locations.to_csv(index=False).encode('utf-8'),
            file_name="top_10_locations.csv",
            mime="text/csv"
        )
        
    elif selected_chart == "15. Top 10 Receiver Cities":
        st.subheader("Top 10 Receiver Cities")
        top_receiver_cities = df_merged['receiver_city'].value_counts().nlargest(10).reset_index()
        top_receiver_cities.columns = ['Receiver City', 'Number of Claims']
        st.bar_chart(top_receiver_cities.set_index('Receiver City'))
        st.download_button(
            label="Download Data as CSV",
            data=top_receiver_cities.to_csv(index=False).encode('utf-8'),
            file_name="top_10_receiver_cities.csv",
            mime="text/csv"
        )
        
    elif selected_chart == "16. Number of Claims by Day of the Month":
        st.subheader("Number of Claims by Day of the Month")
        claims_by_day = df_merged['timestamp'].dt.day.value_counts().sort_index().reset_index()
        claims_by_day.columns = ['Day of Month', 'Number of Claims']
        st.bar_chart(claims_by_day.set_index('Day of Month'))
        st.download_button(
            label="Download Data as CSV",
            data=claims_by_day.to_csv(index=False).encode('utf-8'),
            file_name="claims_by_day_of_month.csv",
            mime="text/csv"
        )
        
    elif selected_chart == "17. Claim Status Distribution by Day of Month (Line Chart)":
        st.subheader("Claim Status Distribution by Day of Month (Line Chart)")
        claims_pivot = df_merged.pivot_table(index=df_merged['timestamp'].dt.day, columns='status', values='claim_id', aggfunc='count', fill_value=0)
        st.line_chart(claims_pivot)
        st.download_button(
            label="Download Data as CSV",
            data=claims_pivot.to_csv().encode('utf-8'),
            file_name="claim_status_by_day.csv",
            mime="text/csv"
        )
        
    elif selected_chart == "18. Claim Status Distribution by Time of Day":
        st.subheader("Claim Status Distribution by Time of Day (Clustered Bar Chart)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_merged, x='time_of_day', hue='status', order=['Morning', 'Afternoon', 'Evening', 'Night'], ax=ax, palette='mako')
        ax.set_title('Claim Status Distribution by Time of Day')
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="Download Chart as PNG",
            data=buf.getvalue(),
            file_name="claim_status_by_time_of_day.png",
            mime="image/png"
        )
        
    elif selected_chart == "19. Number of Claims by Hour of the Day":
        st.subheader("Number of Claims by Hour of the Day")
        claims_by_hour = df_merged['timestamp'].dt.hour.value_counts().sort_index().reset_index()
        claims_by_hour.columns = ['Hour of Day', 'Number of Claims']
        st.bar_chart(claims_by_hour.set_index('Hour of Day'))
        st.download_button(
            label="Download Data as CSV",
            data=claims_by_hour.to_csv(index=False).encode('utf-8'),
            file_name="claims_by_hour_of_day.csv",
            mime="text/csv"
        )
        
    elif selected_chart == "20. Claim Status Distribution by Hour of Day (Line Chart)":
        st.subheader("Claim Status Distribution by Hour of the Day (Line Chart)")
        claims_by_hour_status = df_merged.pivot_table(index=df_merged['timestamp'].dt.hour, columns='status', values='claim_id', aggfunc='count', fill_value=0)
        st.line_chart(claims_by_hour_status)
        st.download_button(
            label="Download Data as CSV",
            data=claims_by_hour_status.to_csv().encode('utf-8'),
            file_name="claim_status_by_hour.csv",
            mime="text/csv"
        )
