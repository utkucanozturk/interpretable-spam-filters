# Interpretable Machine Learning

As shown in previous sections of this report, black box models like SVM and random forest outperformed interpretable algorithms. Therefore, it is necessary to add a layer of interpretability to our analysis. Interpretable machine learning methods will help us deal with opacity of machine learning models and  explain hypotheses about attributes and their connection to the target variable.

From the scammers' point of view, our goal is to improve spam emails so that they pass spam filters. Therefore, the following questions or hypotheses are analyzed with the help of interpretable machine learning methods:

* Which measures about the frequency of words, characters or capital letters in the email should a scammer focus on? Which are the most influential ones?
* Is there a threshold for words, characters or capital letters that the scammer's email should not exceed?
* Are there combinations of words which lead to classification as spam?

More precisely,

* Do many exclamation marks lead to a classification as spam?
* Do many words in capital letters influence the prediction?
* Do some characters influence the model only within certain thresholds?
* Are there strong interactions between finance related words or characters, e.g. `free`, `money`, `credit` or `$`?

In general, our aim is to uncover interesting connections between the features and the target variable. It is also of interest if those connections differ between different model classes. E.g are the same words important for predictions in kernel-based algorithms (like SVMs) and tree-based models (like random forests)?

For this purpose, interpretable machine learning methods such as feature importance, the relationship between target and features (ICE/PDP/ALE), interactions, surrogate models and local models will be introduced and applied to black-box models.

## Feature Importance

Permutation feature importance is, as the name suggests, a method to quantify the importance of a feature. It is measured by calculating the increase in the model’s prediction error after permuting the feature of interest. We measure the prediction error for a specific model with 1-AUC so that a higher AUC leads to a lower error. Since the absolute value is not informative, the ratio between the error of the permuted model and the original model is calculated instead.

Intuitively, permuting or shuffling all values of a feature destroys any relationship between the given feature and the target. If the error increases by breaking the relationship, the given feature must have been important for model prediction.</br></br>

<p align='center' float="left">
<img src="/assets/plots/imp_ranger_pers.png" width="350" /><img src="/assets/plots/imp_ranger_non_pers.png" width="350" />
<img src="/assets/plots/imp_svm_pers.png" width="350" /><img src="/assets/plots/imp_svm_non_pers.png" width="350" />
<figcaption>Permutation Feature Importance for random forest and SVM</figcaption>
</p>

Analyzing the figure above enables us to make statements about the permutation feature importance, such as:

* Permuting the proportion of exclamation marks (`!`) in an email results in a 10 times increase of 1-AUC compared to the original model, when using the personalized random forest model. For non-personalized random forest, the error increases even fifteenfold.
* The top-3 important features (`!`, `remove` and `free`) are the same for the personalized and non-personalized random forest model, as well as for the non-personalized SVM model. Considering the scammers do not know which spam filter is used, this consistency is helpful for setting up rules.
* `hp` is one of the five most important features in the personalized random forest model, with an importance of 4.4. Knowing the recipient's employer is therefore quite important for discriminating between spam and non-spam.
* The personalized words `george` and `hp` are by far the most important features for personalized SVM with a feature importance of 3.6, or 3.4 respectively. If there is the possibility to determine the recipient's name or other personal information it should be mentioned in the email as well.
* The feature importance for most features in random forest is notably higher than for SVM. This might be due to the fact that in random forests splits happen based on specific features, whereas SVMs rely on support vectors, which are specific observations and are therefore made up of all features. However statements on how to improve spam mails cannot be made.

The importance of features is quite consistent whether personalized features are included or not. Therefore, from now on, we will only apply interpretable machine learning methods on the whole dataset, i.e. the dataset including personalized variables.

## Relationship between Target & Features

Now that we know which features are important for the model's prediction, it is now of interest to understand the nature of the impact. Is there a positive or negative influence on the prediction? Is the relationship linear, monotonic, or more complex?
One method for answering these questions is to use partial dependency plots (PDP). They can be used to visualize the relationship between one or two features and the target. The partial dependence function is estimated by:

$$\hat{f}_{x_S}(x_S)=\frac{1}{n}\sum_{i=1}^n\hat{f}(x_S,x^{(i)}_{C})$$

where \(x_S\) are the features we are interested in and all other features are in \(x_C\). We basically average the prediction for a specific value of \(x_S\) over all instances and obtain marginalized effects by this way.

The PDP assumes uncorrelated features, since there would be highly unlikely or even impossible feature combinations considered otherwise. Luckily features which were previously shown to be important show little to no correlations and we can apply this method for further insights.

If you don't plot the average effect over all instances, but instead draw a line for each observation we obtain the individual conditional expectation (ICE). ICE plots show how the instance’s prediction changes when the feature value changes. One advantage of ICE plots compared to PDP plots is that interactions can be uncovered, if the ICE-lines are not parallel, but cross each other.

In figure below you can see that as the number of `!` in an email increases, the marginal spam probability increases for both random forest and SVM. For the random forest model, there are several small jumps in the PDP curve (orange line, e.g., at 0.1 or 0.8), while for SVM it is quite smooth. This could be due to the fact that random forest models are based on certain split points for variables, while SVM is based on the distance of observations to support vectors, which is continuous. This relationship is plausible to the extent that many exclamation marks suggest a sense of urgency, which is a common tactic in spam emails.</br></br>

<p align='center' float="left">
<img src="/assets/plots/pdp_ice_charExclamation_ranger.png" width="350" /><img src="/assets/plots/pdp_ice_charExclamation_svm.png" width="350" />
<figcaption>Centered ICE curves (black) and PDP (orange) for "!"</figcaption>
</p>

On the other hand, the presence of the word `hp` lowers the probability of spam in both the random forest and the SVM (see figure below). This is also plausible, as the knowledge of the recipient's employer shows that the email was sent from a known individual and is not spam sent to multiple people at once. Another explanation might be that the email is sent from within the company and `hp` is in the sender's email signature. In figure below you can see that for random forest there is a steep drop between 0 and 0.5. Above that, the PDP is almost flat. This indicates that the mere mention of `hp` is enough to reduce the spam probability, and flooding the email with the employer's name does not help to pass spam filters.</br></br>

<p align='center' float="left">
<img src="/assets/plots/pdp_ice_hp_ranger.png" width="350" /><img src="/assets/plots/pdp_ice_hp_svm.png" width="350" />
<figcaption>Centered ICE curves (black) and PDP (orange) for "hp"</figcaption>
</p>

As the ICE curves for random forest are all parallel, we do not expect much interaction for the variables `charExclamation` or `hp`. However, if we closely look at the ICE curve for `charExclamation` in the SVM model we can see two different courses:  for some ICE lines the effect continuously increases for increasing values, whereas some lines seem almost flat for increasing values of `charExclamation`. This might indicate interactions, which will further be analyzed later in this report.

To be continued...
