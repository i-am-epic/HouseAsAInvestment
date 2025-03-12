import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy_financial as npf

def calculate_emi(P, annual_interest_rate, years):
    """
    Calculate the fixed monthly EMI for a loan.
    """
    monthly_rate = annual_interest_rate / 100 / 12
    n = years * 12
    EMI = P * monthly_rate * (1 + monthly_rate)**n / ((1 + monthly_rate)**n - 1)
    return EMI

def simulate_investment(params):
    """
    Run a monthly simulation of cash flows and property value evolution.
    The property is broken into a depreciating structure and an appreciating land component.
    Returns a dictionary with arrays of monthly metrics.
    """
    # Basic parameters
    house_price = params['house_price']
    loan_interest = params['loan_interest']
    loan_term = params['loan_term']
    monthly_rent = params['monthly_rent']
    rental_increase = params['rental_increase']
    house_depreciation_rate = params['house_depreciation_rate']
    land_area = params['land_area']
    land_price_per_sqft = params['land_price_per_sqft']
    land_growth_rate = params['land_growth_rate']
    alternative_return = params['alternative_return']
    inflation_rate = params['inflation_rate']
    
    # Advanced parameters (all default to 0 if not changed)
    property_tax_rate = params['property_tax_rate']
    insurance_rate = params['insurance_rate']
    management_fee_rate = params['management_fee_rate']
    maintenance_rate = params['maintenance_rate']
    vacancy_rate = params['vacancy_rate']
    alt_investment_tax = params['alt_investment_tax']
    
    n = int(loan_term * 12)
    EMI = calculate_emi(house_price, loan_interest, loan_term)
    
    # Calculate initial values for land and structure.
    initial_land_value = land_area * land_price_per_sqft
    structure_value_initial = house_price - initial_land_value
    if structure_value_initial < 0:
        raise ValueError("House price is less than calculated land value. Please check your inputs.")
    
    # Initialize arrays for simulation results
    months = np.arange(1, n + 1)
    cumulative_cash_flow = np.zeros(n)
    structure_values = np.zeros(n)
    land_values = np.zeros(n)
    total_property_values = np.zeros(n)
    alt_values = np.zeros(n)
    monthly_net_cash_flow = np.zeros(n)
    monthly_rental_income = np.zeros(n)
    
    # Starting values
    current_rent = monthly_rent
    structure_value = structure_value_initial
    land_value = initial_land_value
    alt_value = house_price  # Initial alternative investment capital
    
    # Monthly factors for depreciation/appreciation and alternative growth
    structure_factor = (1 - house_depreciation_rate / 100) ** (1 / 12)
    land_factor = (1 + land_growth_rate / 100) ** (1 / 12)
    monthly_alt_rate = alternative_return / 100 / 12
    
    # Convert annual expenses to monthly values
    property_tax_monthly = (property_tax_rate / 100 * house_price) / 12
    insurance_monthly = (insurance_rate / 100 * house_price) / 12
    maintenance_monthly = (maintenance_rate / 100 * house_price) / 12

    for m in range(n):
        # Increase rental income at the beginning of each year
        if m > 0 and m % 12 == 0:
            current_rent *= (1 + rental_increase / 100)
        rental_income_effective = current_rent * (1 - vacancy_rate / 100)
        monthly_rental_income[m] = rental_income_effective
        
        # Calculate management fee on rental income
        management_fee = rental_income_effective * (management_fee_rate / 100)
        
        # Net income for the month: effective rent minus EMI and expenses
        net_income = rental_income_effective - EMI - property_tax_monthly - insurance_monthly - management_fee - maintenance_monthly
        monthly_net_cash_flow[m] = net_income
        cumulative_cash_flow[m] = net_income if m == 0 else cumulative_cash_flow[m - 1] + net_income
        
        # Update property values: structure depreciates, land appreciates
        structure_value *= structure_factor
        land_value *= land_factor
        total_value = structure_value + land_value
        
        structure_values[m] = structure_value
        land_values[m] = land_value
        total_property_values[m] = total_value
        
        # Alternative investment grows monthly
        alt_value *= (1 + monthly_alt_rate)
        alt_values[m] = alt_value

    # Apply alternative investment tax on final value
    final_alt_value = alt_values[-1] * (1 - alt_investment_tax / 100)
    alt_values[-1] = final_alt_value

    results = {
        "months": months,
        "cumulative_cash_flow": cumulative_cash_flow,
        "structure_values": structure_values,
        "land_values": land_values,
        "total_property_values": total_property_values,
        "alt_values": alt_values,
        "monthly_net_cash_flow": monthly_net_cash_flow,
        "monthly_rental_income": monthly_rental_income,
        "EMI": EMI
    }
    return results

def generate_yearly_report(results, loan_term, currency):
    """
    Generate a year-by-year DataFrame report from monthly simulation data.
    """
    n = int(loan_term * 12)
    years = np.arange(1, loan_term + 1)
    year_end_indices = [(y * 12 - 1) for y in years]
    
    df = pd.DataFrame({
        "Year": years,
        f"Structure Value ({currency})": results["structure_values"][year_end_indices],
        f"Land Value ({currency})": results["land_values"][year_end_indices],
        f"Total Property Value ({currency})": results["total_property_values"][year_end_indices],
        f"Cumulative Cash Flow ({currency})": results["cumulative_cash_flow"][year_end_indices],
        f"Alternative Investment Value ({currency})": results["alt_values"][year_end_indices],
    })
    df[f"Total Property Benefit ({currency})"] = df[f"Total Property Value ({currency})"] + df[f"Cumulative Cash Flow ({currency})"]
    df[f"Difference (Property Benefit - Alt) ({currency})"] = df[f"Total Property Benefit ({currency})"] - df[f"Alternative Investment Value ({currency})"]
    df["% Difference"] = (df[f"Difference (Property Benefit - Alt) ({currency})"] / df[f"Alternative Investment Value ({currency})"]) * 100
    return df

def calculate_advanced_metrics(cash_flows, discount_rate):
    """
    Calculate advanced financial metrics: NPV and IRR.
    cash_flows: array of monthly cash flows.
    discount_rate: annual discount rate (percent).
    """
    n = len(cash_flows)
    monthly_discount_rate = discount_rate / 100 / 12
    months = np.arange(1, n + 1)
    npv = np.sum(cash_flows / ((1 + monthly_discount_rate) ** months))
    
    irr_monthly = npf.irr(cash_flows)
    irr_annual = (1 + irr_monthly)**12 - 1 if irr_monthly is not None else None
    return npv, irr_annual

def plot_results(results, df_report, currency):
    """
    Plot various graphs for visual analysis.
    Returns a tuple of figures.
    """
    n = len(results["months"])
    
    # Graph 1: Yearly comparison of Total Property Value vs Alternative Investment
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_report["Year"], df_report[f"Total Property Value ({currency})"], label="Total Property Value", marker='o')
    ax1.plot(df_report["Year"], df_report[f"Alternative Investment Value ({currency})"], label="Alternative Investment Value", marker='o', linestyle="--")
    ax1.set_xlabel("Year")
    ax1.set_ylabel(f"Value ({currency})")
    ax1.set_title("Yearly Comparison: Property vs Alternative Investment")
    ax1.legend()
    ax1.grid(True)
    
    # Graph 2: Cumulative Cash Flow over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(results["months"] / 12, results["cumulative_cash_flow"], label="Cumulative Cash Flow", color="purple")
    ax2.set_xlabel("Years")
    ax2.set_ylabel(f"Cumulative Cash Flow ({currency})")
    ax2.set_title("Cumulative Cash Flow Over Time")
    ax2.legend()
    ax2.grid(True)
    
    # Graph 3: Monthly Rental Income vs Net Cash Flow
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(results["months"] / 12, results["monthly_rental_income"], label="Monthly Rental Income", color="green")
    ax3.plot(results["months"] / 12, results["monthly_net_cash_flow"], label="Monthly Net Cash Flow", color="red")
    ax3.set_xlabel("Years")
    ax3.set_ylabel(f"Amount ({currency})")
    ax3.set_title("Monthly Rental Income vs Net Cash Flow")
    ax3.legend()
    ax3.grid(True)
    
    # Graph 4: Yearly % Difference between Property Benefit and Alternative Investment
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(df_report["Year"], df_report["% Difference"], label="% Difference", color="orange", marker='o')
    ax4.set_xlabel("Year")
    ax4.set_ylabel("% Difference")
    ax4.set_title("Yearly % Difference: (Property Benefit - Alternative) / Alternative")
    ax4.legend()
    ax4.grid(True)
    
    return fig1, fig2, fig3, fig4

def main():
    st.title("House Investment Analysis Calculator")
    
    st.markdown("""
    This tool compares buying a house (with structure and land) using a loan versus an alternative investment.
    The report below will show you which investment appears better and provide a structured, year-by-year analysis.
    """)
    
    st.sidebar.header("Input Parameters")
    
    # Basic inputs
    currency = st.sidebar.text_input("Currency Symbol", value="$")
    house_price = st.sidebar.number_input(f"House Structure Price / Loan Amount ({currency})", value=300000, step=10000)
    loan_interest = st.sidebar.number_input("Loan Interest Rate (%)", value=4.0, step=0.1)
    loan_term = st.sidebar.number_input("Loan Term (years)", value=30, step=1)
    monthly_rent = st.sidebar.number_input(f"Initial Monthly Rental Income ({currency})", value=1500, step=50)
    rental_increase = st.sidebar.number_input("Annual Rental Increase (%)", value=3.0, step=0.1)
    house_depreciation_rate = st.sidebar.number_input("House Structure Depreciation Rate (%) per year", value=1.0, step=0.1)
    
    st.sidebar.subheader("Land Details")
    land_area = st.sidebar.number_input("Land Area (sq ft)", value=2000, step=100)
    land_price_per_sqft = st.sidebar.number_input(f"Current Land Price per Sq Ft ({currency})", value=50, step=1)
    land_growth_rate = st.sidebar.number_input("Annual Land Growth Rate (%)", value=3.0, step=0.1)
    
    st.sidebar.subheader("Alternative Investment & Inflation")
    alternative_return = st.sidebar.number_input("Alternative Investment Return Rate (%)", value=5.0, step=0.1)
    inflation_rate = st.sidebar.number_input("Inflation Rate (%)", value=2.0, step=0.1)
    discount_rate = st.sidebar.number_input("Discount Rate for NPV Calculation (%)", value=5.0, step=0.1)
    
    # Advanced parameters hidden behind an expander (all default to 0)
    with st.sidebar.expander("Advanced Options (default = 0)"):
        property_tax_rate = st.number_input("Annual Property Tax Rate (%)", value=0.0, step=0.1)
        insurance_rate = st.number_input("Annual Insurance Rate (%)", value=0.0, step=0.1)
        management_fee_rate = st.number_input("Management Fee Rate (%) on Rental Income", value=0.0, step=0.1)
        maintenance_rate = st.number_input("Annual Maintenance/Repair Rate (%) of House Price", value=0.0, step=0.1)
        vacancy_rate = st.number_input("Vacancy Rate (%)", value=0.0, step=0.1)
        alt_investment_tax = st.number_input("Alternative Investment Tax (%)", value=0.0, step=0.1)
    
    run_simulation = st.sidebar.button("Generate Report")
    
    if run_simulation:
        params = {
            'house_price': house_price,
            'loan_interest': loan_interest,
            'loan_term': loan_term,
            'monthly_rent': monthly_rent,
            'rental_increase': rental_increase,
            'house_depreciation_rate': house_depreciation_rate,
            'land_area': land_area,
            'land_price_per_sqft': land_price_per_sqft,
            'land_growth_rate': land_growth_rate,
            'alternative_return': alternative_return,
            'inflation_rate': inflation_rate,
            'property_tax_rate': property_tax_rate,
            'insurance_rate': insurance_rate,
            'management_fee_rate': management_fee_rate,
            'maintenance_rate': maintenance_rate,
            'vacancy_rate': vacancy_rate,
            'alt_investment_tax': alt_investment_tax
        }
        
        try:
            results = simulate_investment(params)
        except ValueError as e:
            st.error(str(e))
            return
        
        df_report = generate_yearly_report(results, loan_term, currency)
        
        # Summary figures
        n = int(loan_term * 12)
        EMI = results["EMI"]
        final_structure_value = results["structure_values"][-1]
        final_land_value = results["land_values"][-1]
        final_total_property_value = results["total_property_values"][-1]
        final_cumulative_cash_flow = results["cumulative_cash_flow"][-1]
        final_alt_value = results["alt_values"][-1]
        final_total_property_benefit = final_total_property_value + final_cumulative_cash_flow
        
        st.subheader("Summary Report")
        st.write(f"**Monthly EMI:** {currency}{EMI:,.2f}")
        st.write(f"**Total Payment over {loan_term} years:** {currency}{EMI * n:,.2f}")
        st.write(f"**Final House Structure Value:** {currency}{final_structure_value:,.2f}")
        st.write(f"**Final Land Value:** {currency}{final_land_value:,.2f}")
        st.write(f"**Final Total Property Value (Structure + Land):** {currency}{final_total_property_value:,.2f}")
        st.write(f"**Cumulative Cash Flow from Renting:** {currency}{final_cumulative_cash_flow:,.2f}")
        st.write(f"**Total Benefit from Property (Value + Cash Flow):** {currency}{final_total_property_benefit:,.2f}")
        st.write(f"**Alternative Investment Value (after tax):** {currency}{final_alt_value:,.2f}")
        
        # Indicate which investment is better
        if final_total_property_benefit > final_alt_value:
            st.success("**Investment Analysis:** Buying the house appears to be the better investment.")
        else:
            st.error("**Investment Analysis:** Investing the money elsewhere appears to be the better option.")
        
        # Calculate advanced metrics
        npv, irr_annual = calculate_advanced_metrics(results["monthly_net_cash_flow"], discount_rate)
        st.write(f"**NPV of Cash Flows (using {discount_rate}% discount rate):** {currency}{npv:,.2f}")
        if irr_annual is not None:
            st.write(f"**IRR (Annualized):** {irr_annual * 100:.2f}%")
        else:
            st.write("IRR could not be calculated.")
        
        # Display Yearly Report Table
        st.subheader("Yearly Investment Report")
        st.dataframe(df_report.style.format({
            f"Structure Value ({currency})": "{:,.2f}",
            f"Land Value ({currency})": "{:,.2f}",
            f"Total Property Value ({currency})": "{:,.2f}",
            f"Cumulative Cash Flow ({currency})": "{:,.2f}",
            f"Alternative Investment Value ({currency})": "{:,.2f}",
            f"Total Property Benefit ({currency})": "{:,.2f}",
            f"Difference (Property Benefit - Alt) ({currency})": "{:,.2f}",
            "% Difference": "{:.2f}%"
        }))
        
        # Generate and display graphs
        fig1, fig2, fig3, fig4 = plot_results(results, df_report, currency)
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
        st.pyplot(fig4)
        
        st.markdown("### Analysis and Discussion")
        st.markdown(f"""
        **Overview:**  
        This report provides a structured, year-by-year analysis of your property investment versus an alternative investment.
        
        **Key Components:**  
        - **House Structure:** Depreciates over time based on the input rate.  
        - **Land:** Appreciates over time as per the given growth rate.  
        - **Rental Income:** Adjusted for vacancy and reduced by expenses such as property tax, insurance, management fees, and maintenance.
        
        **Financial Metrics:**  
        - **Cumulative Cash Flow:** Aggregates the net monthly income (which may be negative in some periods).  
        - **Total Property Benefit:** Sum of the property’s market value and the cumulative cash flow.  
        - **NPV & IRR:** These advanced metrics help evaluate the investment's return considering the time value of money.
        
        **Investment Decision:**  
        Compare the Total Property Benefit with the Alternative Investment Value (after tax).  
        In this analysis, **{'buying the house' if final_total_property_benefit > final_alt_value else 'investing elsewhere'}** appears to be the better investment option.
        """)
    
if __name__ == '__main__':
    main()
