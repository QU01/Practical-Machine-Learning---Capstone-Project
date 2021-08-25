# **Course Project**

## **Background**
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## **Data**

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

## **What you should submit**

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

## **Peer Review Portion**

Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).

## **Course Project Prediction Quiz Portion**

Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading. 

## **Reproducibility**

Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 

## **Exploratory Data Analysis**


```R
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(gbm)
```

    Loading required package: lattice
    
    Loading required package: ggplot2
    
    randomForest 4.6-14
    
    Type rfNews() to see new features/changes/bug fixes.
    
    
    Attaching package: ‘randomForest’
    
    
    The following object is masked from ‘package:ggplot2’:
    
        margin
    
    
    Loaded gbm 2.1.8
    



```R
training_data <- read.csv('pml-training.csv')
testing_data <- read.csv('pml-testing.csv')
```


```R
head(training_data)
```


<table class="dataframe">
<caption>A data.frame: 6 × 160</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>user_name</th><th scope=col>raw_timestamp_part_1</th><th scope=col>raw_timestamp_part_2</th><th scope=col>cvtd_timestamp</th><th scope=col>new_window</th><th scope=col>num_window</th><th scope=col>roll_belt</th><th scope=col>pitch_belt</th><th scope=col>yaw_belt</th><th scope=col>⋯</th><th scope=col>gyros_forearm_x</th><th scope=col>gyros_forearm_y</th><th scope=col>gyros_forearm_z</th><th scope=col>accel_forearm_x</th><th scope=col>accel_forearm_y</th><th scope=col>accel_forearm_z</th><th scope=col>magnet_forearm_x</th><th scope=col>magnet_forearm_y</th><th scope=col>magnet_forearm_z</th><th scope=col>classe</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>carlitos</td><td>1323084231</td><td>788290</td><td>05/12/2011 11:23</td><td>no</td><td>11</td><td>1.41</td><td>8.07</td><td>-94.4</td><td>⋯</td><td>0.03</td><td> 0.00</td><td>-0.02</td><td>192</td><td>203</td><td>-215</td><td>-17</td><td>654</td><td>476</td><td>A</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>carlitos</td><td>1323084231</td><td>808298</td><td>05/12/2011 11:23</td><td>no</td><td>11</td><td>1.41</td><td>8.07</td><td>-94.4</td><td>⋯</td><td>0.02</td><td> 0.00</td><td>-0.02</td><td>192</td><td>203</td><td>-216</td><td>-18</td><td>661</td><td>473</td><td>A</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>carlitos</td><td>1323084231</td><td>820366</td><td>05/12/2011 11:23</td><td>no</td><td>11</td><td>1.42</td><td>8.07</td><td>-94.4</td><td>⋯</td><td>0.03</td><td>-0.02</td><td> 0.00</td><td>196</td><td>204</td><td>-213</td><td>-18</td><td>658</td><td>469</td><td>A</td></tr>
	<tr><th scope=row>4</th><td>4</td><td>carlitos</td><td>1323084232</td><td>120339</td><td>05/12/2011 11:23</td><td>no</td><td>12</td><td>1.48</td><td>8.05</td><td>-94.4</td><td>⋯</td><td>0.02</td><td>-0.02</td><td> 0.00</td><td>189</td><td>206</td><td>-214</td><td>-16</td><td>658</td><td>469</td><td>A</td></tr>
	<tr><th scope=row>5</th><td>5</td><td>carlitos</td><td>1323084232</td><td>196328</td><td>05/12/2011 11:23</td><td>no</td><td>12</td><td>1.48</td><td>8.07</td><td>-94.4</td><td>⋯</td><td>0.02</td><td> 0.00</td><td>-0.02</td><td>189</td><td>206</td><td>-214</td><td>-17</td><td>655</td><td>473</td><td>A</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>carlitos</td><td>1323084232</td><td>304277</td><td>05/12/2011 11:23</td><td>no</td><td>12</td><td>1.45</td><td>8.06</td><td>-94.4</td><td>⋯</td><td>0.02</td><td>-0.02</td><td>-0.03</td><td>193</td><td>203</td><td>-215</td><td> -9</td><td>660</td><td>478</td><td>A</td></tr>
</tbody>
</table>




```R
head(testing_data)
```


<table class="dataframe">
<caption>A data.frame: 6 × 160</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>user_name</th><th scope=col>raw_timestamp_part_1</th><th scope=col>raw_timestamp_part_2</th><th scope=col>cvtd_timestamp</th><th scope=col>new_window</th><th scope=col>num_window</th><th scope=col>roll_belt</th><th scope=col>pitch_belt</th><th scope=col>yaw_belt</th><th scope=col>⋯</th><th scope=col>gyros_forearm_x</th><th scope=col>gyros_forearm_y</th><th scope=col>gyros_forearm_z</th><th scope=col>accel_forearm_x</th><th scope=col>accel_forearm_y</th><th scope=col>accel_forearm_z</th><th scope=col>magnet_forearm_x</th><th scope=col>magnet_forearm_y</th><th scope=col>magnet_forearm_z</th><th scope=col>problem_id</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>pedro </td><td>1323095002</td><td>868349</td><td>05/12/2011 14:23</td><td>no</td><td> 74</td><td>123.00</td><td> 27.00</td><td> -4.75</td><td>⋯</td><td> 0.74</td><td>-3.34</td><td>-0.59</td><td>-110</td><td>267</td><td>-149</td><td>-714</td><td> 419</td><td>617</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>2</td><td>jeremy</td><td>1322673067</td><td>778725</td><td>30/11/2011 17:11</td><td>no</td><td>431</td><td>  1.02</td><td>  4.87</td><td>-88.90</td><td>⋯</td><td> 1.12</td><td>-2.78</td><td>-0.18</td><td> 212</td><td>297</td><td>-118</td><td>-237</td><td> 791</td><td>873</td><td>2</td></tr>
	<tr><th scope=row>3</th><td>3</td><td>jeremy</td><td>1322673075</td><td>342967</td><td>30/11/2011 17:11</td><td>no</td><td>439</td><td>  0.87</td><td>  1.82</td><td>-88.50</td><td>⋯</td><td> 0.18</td><td>-0.79</td><td> 0.28</td><td> 154</td><td>271</td><td>-129</td><td> -51</td><td> 698</td><td>783</td><td>3</td></tr>
	<tr><th scope=row>4</th><td>4</td><td>adelmo</td><td>1322832789</td><td>560311</td><td>02/12/2011 13:33</td><td>no</td><td>194</td><td>125.00</td><td>-41.60</td><td>162.00</td><td>⋯</td><td> 1.38</td><td> 0.69</td><td> 1.80</td><td> -92</td><td>406</td><td> -39</td><td>-233</td><td> 783</td><td>521</td><td>4</td></tr>
	<tr><th scope=row>5</th><td>5</td><td>eurico</td><td>1322489635</td><td>814776</td><td>28/11/2011 14:13</td><td>no</td><td>235</td><td>  1.35</td><td>  3.33</td><td>-88.60</td><td>⋯</td><td>-0.75</td><td> 3.10</td><td> 0.80</td><td> 131</td><td>-93</td><td> 172</td><td> 375</td><td>-787</td><td> 91</td><td>5</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>jeremy</td><td>1322673149</td><td>510661</td><td>30/11/2011 17:12</td><td>no</td><td>504</td><td> -5.92</td><td>  1.59</td><td>-87.70</td><td>⋯</td><td>-0.88</td><td> 4.26</td><td> 1.35</td><td> 230</td><td>322</td><td>-144</td><td>-300</td><td> 800</td><td>884</td><td>6</td></tr>
</tbody>
</table>




```R
dim(training_data)
dim(testing_data)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>19622</li><li>160</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20</li><li>160</li></ol>



## **Preprocessing**


```R
nonZero <- nearZeroVar(training_data)

training_data <- training_data[, -nonZero]
testing_data <- testing_data[, -nonZero]
```


```R
dim(training_data)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>19622</li><li>100</li></ol>




```R
training_data <- training_data[,nas == FALSE]
testing_data <- testing_data[,nas == FALSE]
```


```R
dim(training_data)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>19622</li><li>59</li></ol>




```R
sapply(training_data, function(x) is.character(x))
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>raw_timestamp_part_1</dt><dd>FALSE</dd><dt>raw_timestamp_part_2</dt><dd>FALSE</dd><dt>num_window</dt><dd>FALSE</dd><dt>roll_belt</dt><dd>FALSE</dd><dt>pitch_belt</dt><dd>FALSE</dd><dt>yaw_belt</dt><dd>FALSE</dd><dt>total_accel_belt</dt><dd>FALSE</dd><dt>gyros_belt_x</dt><dd>FALSE</dd><dt>gyros_belt_y</dt><dd>FALSE</dd><dt>gyros_belt_z</dt><dd>FALSE</dd><dt>accel_belt_x</dt><dd>FALSE</dd><dt>accel_belt_y</dt><dd>FALSE</dd><dt>accel_belt_z</dt><dd>FALSE</dd><dt>magnet_belt_x</dt><dd>FALSE</dd><dt>magnet_belt_y</dt><dd>FALSE</dd><dt>magnet_belt_z</dt><dd>FALSE</dd><dt>roll_arm</dt><dd>FALSE</dd><dt>pitch_arm</dt><dd>FALSE</dd><dt>yaw_arm</dt><dd>FALSE</dd><dt>total_accel_arm</dt><dd>FALSE</dd><dt>gyros_arm_x</dt><dd>FALSE</dd><dt>gyros_arm_y</dt><dd>FALSE</dd><dt>gyros_arm_z</dt><dd>FALSE</dd><dt>accel_arm_x</dt><dd>FALSE</dd><dt>accel_arm_y</dt><dd>FALSE</dd><dt>accel_arm_z</dt><dd>FALSE</dd><dt>magnet_arm_x</dt><dd>FALSE</dd><dt>magnet_arm_y</dt><dd>FALSE</dd><dt>magnet_arm_z</dt><dd>FALSE</dd><dt>roll_dumbbell</dt><dd>FALSE</dd><dt>pitch_dumbbell</dt><dd>FALSE</dd><dt>yaw_dumbbell</dt><dd>FALSE</dd><dt>total_accel_dumbbell</dt><dd>FALSE</dd><dt>gyros_dumbbell_x</dt><dd>FALSE</dd><dt>gyros_dumbbell_y</dt><dd>FALSE</dd><dt>gyros_dumbbell_z</dt><dd>FALSE</dd><dt>accel_dumbbell_x</dt><dd>FALSE</dd><dt>accel_dumbbell_y</dt><dd>FALSE</dd><dt>accel_dumbbell_z</dt><dd>FALSE</dd><dt>magnet_dumbbell_x</dt><dd>FALSE</dd><dt>magnet_dumbbell_y</dt><dd>FALSE</dd><dt>magnet_dumbbell_z</dt><dd>FALSE</dd><dt>roll_forearm</dt><dd>FALSE</dd><dt>pitch_forearm</dt><dd>FALSE</dd><dt>yaw_forearm</dt><dd>FALSE</dd><dt>total_accel_forearm</dt><dd>FALSE</dd><dt>gyros_forearm_x</dt><dd>FALSE</dd><dt>gyros_forearm_y</dt><dd>FALSE</dd><dt>gyros_forearm_z</dt><dd>FALSE</dd><dt>accel_forearm_x</dt><dd>FALSE</dd><dt>accel_forearm_y</dt><dd>FALSE</dd><dt>accel_forearm_z</dt><dd>FALSE</dd><dt>magnet_forearm_x</dt><dd>FALSE</dd><dt>magnet_forearm_y</dt><dd>FALSE</dd><dt>magnet_forearm_z</dt><dd>FALSE</dd><dt>classe</dt><dd>TRUE</dd></dl>




```R
chars <- sapply(training_data, is.character)
```


```R
training_data <- subset(training_data, select = -c(X, user_name, cvtd_timestamp))
```

### **Models**


```R
model <- train(classe ~ ., data = training_data, method="rpart")
```


```R
pred <- predict(model, testing_data)
```


```R
confusionMatrix(pred, as.factor(testing_data$classe))$overall["Accuracy"]
```


<strong>Accuracy:</strong> 0.998301486199575



```R
model_2 <- train(classe ~ ., data = training_data, method = "rf", ntree = 50)
```


```R
pred1 <- predict(model_2, testing_data)
confusionMatrix(pred1, as.factor(testing_data$classe))$overall["Accuracy"]
```


<strong>Accuracy:</strong> 1



```R
model_3 <- train(classe ~ ., data = training_data, method = "gbm", verbose = FALSE)
```


```R
pred2 <- predict(model_3, testing_data)
confusionMatrix(pred2, as.factor(testing_data$classe))$overall["Accuracy"]
```


<strong>Accuracy:</strong> 0.998301486199575


## **Conclusion**

The Random Forest Model has a perfect accuracy of 1, so we will use this model for the final prediction


```R
final_pred <- read.csv('pml-testing.csv')
```


```R
predict(model_2, final_pred)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>B</li><li>A</li><li>B</li><li>A</li><li>A</li><li>E</li><li>D</li><li>B</li><li>A</li><li>A</li><li>B</li><li>C</li><li>B</li><li>A</li><li>E</li><li>E</li><li>A</li><li>B</li><li>B</li><li>B</li></ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<style>
	.list-inline {list-style: none; margin:0; padding: 0}
	.list-inline>li {display: inline-block}
	.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
	</style>
	<ol class=list-inline><li>'A'</li><li>'B'</li><li>'C'</li><li>'D'</li><li>'E'</li></ol>
</details>

