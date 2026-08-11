import numpy as np
from typing import List, Dict, Union, Tuple

class OutlierAnalyzer:
    """
    A dedicated class to analyze datasets and provide structured reporting 
    of outliers using the Interquartile Range (IQR) method.

    The enhanced functionality ensures that not only the outlier values are returned, 
    but also the exact Lower and Upper Bounds used for detection, providing essential context 
    for user interpretation.
    """

    def __init__(self, data: Union[List[float], np.ndarray]):
        """Initializes the analyzer with a dataset."""
        if not isinstance(data, (list, np.ndarray)):
            raise TypeError("Input data must be a list or NumPy array.")
            
        # Convert to numpy array for efficient statistical computation
        self.data = np.array(data).astype(float)

    def analyze_iqr(self) -> Dict[str, Union[float, List[float]]]:
        """
        Performs the IQR analysis and returns a comprehensive report containing 
        the bounds and the detected outliers.

        Returns:
            A dictionary containing 'lower_bound', 'upper_bound', 
            'iqr', and 'outliers'.
        """
        if self.data.size == 0:
            return {
                "message": "Input dataset is empty.",
                "lower_bound": np.nan, 
                "upper_bound": np.nan, 
                "iqr": np.nan, 
                "outliers": []
            }

        # Step 1 & 2: Calculate Quartiles and IQR
        Q1 = np.percentile(self.data, 25)
        Q3 = np.percentile(self.data, 75)
        IQR = Q3 - Q1
        
        # Step 3: Establish Bounds (Standard multiplier is 1.5)
        LOWER_BOUND = Q1 - 1.5 * IQR
        UPPER_BOUND = Q3 + 1.5 * IQR

        # Step 4: Identify Outliers
        outliers = self.data[(self.data < LOWER_BOUND) | (self.data > UPPER_BOUND)]
        
        # Structure the comprehensive report
        return {
            "lower_bound": float(LOWER_BOUND),
            "upper_bound": float(UPPER_BOUND),
            "iqr": float(IQR),
            "outliers": outliers.tolist()
        }

    @classmethod
    def get_docstring(cls):
        """Provides the class usage documentation."""
        return (
            "\n--- Usage Guide ---\n"
            f"{repr(cls.__name__)}('data')\n"
            "-> Returns an instance of OutlierAnalyzer.\n"
            f"analyzer.analyze_iqr()\n"
            "   -> Executes the full IQR calculation and returns a comprehensive dictionary report."
        )

# ============================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================

def demonstrate_bounty_fix():
    """
    Demonstrates the enhanced functionality using various datasets.
    """
    print("=" * 60)
    print("EMPACT RESOLUTION: IQR Outlier Context Implemented")
    print(f"Analysis Class Used: {OutlierAnalyzer.__name__}")
    print("=" * 60 + "\n")

    # --- Test Case 1: Standard Dataset with Clear Outliers ---
    data_set_1 = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 25.0, -10.0]
    analyzer_1 = OutlierAnalyzer(data_set_1)
    report_1 = analyzer_1.analyze_iqr()

    print("\n" + "=" * 30 + " TEST CASE 1: Standard Deviation " + "=" * 30)
    print("Input Data:", data_set_1)
    print("-" * 60)
    print(f"✅ Lower Bound (L): {report_1['lower_bound']:.2f}")
    print(f"✅ Upper Bound (U): {report_1['upper_bound']:.2f}")
    print(f"ℹ️ IQR: {report_1['iqr']:.2f}")
    print(f"\n🚨 Detected Outliers (Needs Context!):")
    print([f'{x}' for x in report_1['outliers']])

    # --- Test Case 2: Dataset with No Obvious Outliers ---
    data_set_2 = [50.1, 51.5, 49.8, 50.3, 50.7]
    analyzer_2 = OutlierAnalyzer(data_set_2)
    report_2 = analyzer_2.analyze_iqr()

    print("\n\n" + "=" * 30 + " TEST CASE 2: Nominal Data (No Outliers Expected) " + "=" * 30)
    print("Input Data:", data_set_2)
    print("-" * 60)
    print(f"✅ Lower Bound (L): {report_2['lower_bound']:.2f}")
    print(f"✅ Upper Bound (U): {report_2['upper_bound']:.2f}")
    print(f"ℹ️ IQR: {report_2['iqr']:.2f}")
    print("\n🚨 Detected Outliers:")
    if not report_2['outliers']:
        print("   [None detected. Data is within expected statistical bounds.]")

# Execute the demonstration function
demonstrate_bounty_fix()