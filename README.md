<p align="center">
  <a href="#">
    <img src="https://badges.pufler.dev/visits/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Visits Badge">
    <img src="https://badges.pufler.dev/updated/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Updated Badge">
    <img src="https://badges.pufler.dev/created/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Created Badge">
    <img src="https://img.shields.io/github/contributors/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Contributors Badge">
    <img src="https://img.shields.io/github/last-commit/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Last Commit Badge">
    <img src="https://img.shields.io/github/commit-activity/m/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Commit Activity Badge">
    <img src="https://img.shields.io/github/repo-size/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="Repo Size Badge">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="Contributions welcome">
    <img src="https://www.codefactor.io/repository/github/azaryasph/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning" alt="CodeFactor" />
  </a>
</p>

# <img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30"> Mini Project 4: Predict Customer Clicked Ads Classification <img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30">

<p align="center">
    <img src="https://images.unsplash.com/photo-1541535650810-10d26f5c2ab3?q=80&w=1776&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" width=600 height=320>
</p>

Photo by <a href="https://unsplash.com/@anthonyrosset?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Anthony Rosset</a> on <a href="https://unsplash.com/photos/group-of-people-walking-near-high-rise-buildings-5r5554u-mHo?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  
## Table of Contents
1. [Background Project](#Bacgkrpund-Project)
2. [Scope of Work](#Scope-of-Work)
3. [Data and Assumptions](#Data-and-Assumptions)
4. [Data Analysis](#Data-Analysis)
5. [Data Preprocessing](#Data-Preprocessing)
5. [Modelling and Evaluation](#Modelling-and-Evaluation)
6. [Conclusion](#Conclusion)
7. [Suggestions](#Suggestions)
8. [Installation and Usage](#Installation-and-Usage)

## Background Project
### What is the problem to solve
The problem is the company wants to **target the right customers who are more likely to click on the ads** so the **company don't waste their money** on giving advertisements to the wrong customers.

### Why is this problem important?
This problem is important **because the company can save a lot of money by targeting the right customers** who are more likely to click on the ads. This will help the company to increase their profit.

### What is the goal of this project?
The goal of this project is to **increase the conversion rate** of the company and also **will help the company to increase their profit**.

### What Will be Changed if There are the Model Results?
If there are the model results, the company will **target the right customers who are more likely to click on the ads** so the company **don't waste their money** on giving advertisements to the wrong customers.

## Scope of Work
### What is the scope of this project?
The scope of this project is to **predict the customers who are more likely to click on the ads** so the company can **target the right customers**.

### How is the output of the developed model?
The output of the developed model is the classification prediction of the customers who are more likely to click on the ads and not.

## Data & Assumption
### Data Size
The data size is 1000 rows and 11 columns.

### Features and Description
Features | Description
--- | ---
`Unnamed: 0` | This is an index column.
`daily_time_spent_on_site` | This is a numerical field that represents the amount of time a customer spends on the site daily. It's measured in minutes. 
`age` | This is a numerical field that represents the age of the customer.
`area_income` | This is a numerical field that represents the income of the area where the user lives. It's measured in Indonesian Rupiah.
`daily_internet_usage` | This is a numerical field that represents the amount of time a user spends on the internet daily. It's measured in minutes.
`gender` | This is a categorical field that represents the gender of the user. It's in Indonesian, with "Perempuan" meaning female and "Laki-Laki" meaning male.
`timestamp` | This is a datetime field that represents when the user clicked on the ad. It's in the format of month/day/year hour:minute.
`clicked_on_ad` | This is a categorical field that represents whether the user clicked on the ad. "No" means the user did not click on the ad.
`city` | This is a categorical field that represents the city where the user lives.
`province` | This is a categorical field that represents the province where the user lives.
`ad_category` | This is a categorical field that represents the category of the ad.

### Assumptions
Based on my domain knowledge, I assume that the feature that will be useful for predicting whether the customer will click on the ads or not are:
- `daily_time_spent_on_site`
- `age`
- `area_income` / `income`
- `daily_internet_usage`
- `city`
- `province`
- `ad_category`

The rest of the features are not useful for this prediction task, especially timestamp feature this feature is data leakage and this feature will only be used for Data Analysis only.


