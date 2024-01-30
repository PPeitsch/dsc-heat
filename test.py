import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def generate_extrapolated_lines(x_values, y_values, peak_start, peak_end):
    """
    Generate extrapolated lines before and after the peak.

    :param x_values: Array of x values.
    :param y_values: Array of y values.
    :param peak_start: Index of the start of the peak.
    :param peak_end: Index of the end of the peak.
    :return: Tuple of two arrays representing extrapolated lines (before, after).
    """
    # Calculate slopes of the lines before and after the peak
    slope_before = (y_values[peak_start] - y_values[peak_start - 1]) / (x_values[peak_start] - x_values[peak_start - 1])
    slope_after = (y_values[peak_end + 1] - y_values[peak_end]) / (x_values[peak_end + 1] - x_values[peak_end])

    # Calculate intercepts of the lines
    intercept_before = y_values[peak_start] - slope_before * x_values[peak_start]
    intercept_after = y_values[peak_end] - slope_after * x_values[peak_end]

    # Generate x values for extrapolation
    x_before = np.linspace(x_values[0],
                           x_values[peak_start - 1],
                           num=peak_start)
    x_after = np.linspace(x_values[peak_end + 1],
                          x_values[-1],
                          num=len(x_values) - peak_end - 1)

    # Calculate y values for extrapolation
    y_before = slope_before * x_before + intercept_before
    y_after = slope_after * x_after + intercept_after

    return y_before, y_after


def calculate_alpha_curve(x_values, y_values, baseline_curve, peak_start: int, peak_end: int):
    """
    Calculate the alpha curve for a given baseline.

    :param x_values: Array of x values.
    :param y_values: Array of y values.
    :param baseline_curve: Array representing the baseline curve.
    :param peak_start: Index of peak start.
    :param peak_end: Index of peak end.
    :return: Array representing the alpha curve.
    """

    f = y_values[peak_start:peak_end + 1] - baseline_curve[peak_start:peak_end + 1]
    print(f"alpha = G - F (baseline) \nalpha: {f}, type: {type(f)}")
    # Calculate the integrals for alpha curve
    sum_integral = np.sum(np.trapz(f, x_values[peak_start:peak_end + 1]))
    print(f"integral of alpha: {sum_integral}, type: {type(sum_integral)}")

    # Avoid division by zero
    if sum_integral != 0:
        print(f"G - F: \nG = {y_values[peak_start:peak_end + 1]} \nF = {baseline_curve[peak_start:peak_end + 1]}")
        # Calculate alpha curve using the cumulative integral
        alpha_curve = np.divide(np.cumsum(y_values[peak_start:peak_end + 1] - baseline_curve[peak_start:peak_end + 1]),
                                sum_integral)
    else:
        # Set alpha_curve to zeros if sum_integral is zero
        alpha_curve = np.zeros_like(baseline_curve[peak_start:peak_end + 1])

    return alpha_curve


def calculate_baseline(x_values, y_values, peak_start, peak_end, iterations: int = 50):
    """
    Calculate the baseline for a given curve using extrapolation and iteration.

    :param x_values: Array of x values.
    :param y_values: Array of y values.
    :param peak_start: Index of the start of the peak.
    :param peak_end: Index of the end of the peak.
    :param iterations: Number of iterations for the recursive algorithm.
    :return: Array of baseline y values.
    """
    # Generate extrapolated lines before and after the peak
    extrapolated_lines_before, extrapolated_lines_after = generate_extrapolated_lines(x_values,
                                                                                      y_values,
                                                                                      peak_start,
                                                                                      peak_end)
    print(f"extrapolated line before: {extrapolated_lines_before}")
    print(f"extrapolated line after:{extrapolated_lines_after}")

    # Initialize function with the initial baseline estimate
    function = np.concatenate((extrapolated_lines_before,
                               y_values[peak_start:peak_end + 1],
                               extrapolated_lines_after))

    print(f"F curve: {function}")

    for n in range(iterations):
        # Calculate alpha curve
        alpha_curve = calculate_alpha_curve(x_values,
                                            y_values,
                                            function,
                                            peak_start,
                                            peak_end)

        # Interpolate alpha_curve to have the same length as y_values[peak_start:peak_end + 1]
        interp_alpha_curve = interp1d(np.linspace(0, 1, len(alpha_curve)),
                                      alpha_curve,
                                      kind='linear')(np.linspace(0, 1, len(y_values[peak_start:peak_end + 1])))

        print(f"F curve: {function}")
        print(f"alpha cumulative integral / integral: {alpha_curve}")
        print(f"interpolated curve: {interp_alpha_curve}")

        function[peak_start:peak_end + 1] = interp_alpha_curve

        """
        # Update function using the interpolated alpha_curve and the extrapolated lines
        function[peak_start:peak_end + 1] = interp_alpha_curve * (
                extrapolated_lines_after[:-1] - extrapolated_lines_before) + extrapolated_lines_before
        """

    return function, extrapolated_lines_before, extrapolated_lines_after

# Example usage:
# Assuming x_values and y_values are your data arrays, and peak_start, peak_end are indices of the peak
# You can replace these values with your actual data
x_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_values = np.array([0, 0, 1, 2, 4, 5, 6, 7, 9, 11, 12,
                     12, 11, 9, 7, 6, 5, 3, 2, 1, 0])
peak_start = 4
peak_end = 16

baseline_curve, extrapolated_lines_before, extrapolated_lines_after = calculate_baseline(x_values,
                                                                                         y_values,
                                                                                         peak_start,
                                                                                         peak_end)

# Plotting the original curve, baseline curve, and extrapolated lines
plt.plot(x_values,
         y_values,
         label='Original Curve')
plt.plot(x_values[peak_start:peak_end + 1],
         baseline_curve[peak_start:peak_end + 1],
         label='Baseline Curve',
         linestyle='--',
         color='red')
plt.plot(x_values[:peak_start],
         extrapolated_lines_before,
         label='Extrapolated Lines Before',
         linestyle='--',
         color='green')
plt.plot(x_values[peak_end + 1:],
         extrapolated_lines_after,
         label='Extrapolated Lines After',
         linestyle='--',
         color='blue')
plt.legend()
plt.show()

print(baseline_curve)