## LAD Anomaly Detection Model

Implementation of **Anomaly Detection for High-Dimensional Data Using Large Deviations Principle**: _Sreelekha Guggilam and Varun Chandola and Abani Patra, Anomaly Detection for High-Dimensional Data Using Large Deviations Principle(In preparation)(2021)._


![Image](https://github.com/sreelekh/LAD/blob/main/Confirmed_total_full_history_combined_top_10_per_capita.png)

![Image](https://github.com/sreelekh/LAD/blob/main/Deaths_total_full_history_combined_top_10_per_capita.png)

Top 5 anomalous counties identified by the proposed LAD algorithm based on the daily multivariate time-series, consisting ofcumulative COVID-19 per-capita infections and deaths. At any time-instance, the algorithm analyzes the bi-variate time series for all thecounties to identify anomalies. The time-series for the non-anomalous counties are plotted (light-gray) in the background for reference. For the counties in North Dakota (Burleigh and Grand Forks), the number of confirmed cases (top), and the sharp rise in November 2020, is theprimary cause for anomaly. On the other hand, Wayne County in Michigan was identified as anomalous primarily because of its abnormallyhigh death rate, especially when compared to the relatively moderate confirmed infection rate.


## File Descriptions
1. Run import_libraries.ipynb, import_functions.ipynb, import_global_params.ipynb (optional) to import required libraries and functions
4. Run LDP_paper_results_8-Evaluation Small, large.ipynb to run the LAD model on datasets
5. Run LDP_paper_results_8-COVID TS plots only 50k population lower limit.ipynb to generate plots for COVID-19 data for US Counties
