import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.mlab as mlab
import seaborn as sns
import matplotlib.pylab as plt
sns.set(style='white')

sem = lambda s, n: s / np.sqrt(n)
zstat = lambda xbar, mu, sigma, n: (xbar - mu) / sem(sigma, n)
tstat = lambda xbar, mu, s, n: (xbar - mu)/sem(s, n)
tconf = lambda x, tcrit, s, n: (x - tcrit * sem(s, n), x + tcrit * sem(s, n))
zconf = lambda x, zcrit, s, n: (x - zcrit * sem(s, n), x + zcrit * sem(s, n))
ss = lambda s1, n1, s2, n2: np.sqrt(s1**2.0/n1 + s2**2.0/n2)
spool = lambda s1, df1, s2, df2: np.sqrt((s1 + s2) / df1 + df2)
cd = lambda mu, s: mu / s
dftmin = lambda n1, n2: min([n1, n2])
dft = lambda n1, n2: n1 + n2 - 2
df = lambda n: n -1
r2 = lambda t, df: t**2.0 / (t**2.0 + df)

background_info = "In a Stroop task, participants are presented with a list of words," \
                  " with each word displayed in a color of ink. The participants task " \
                  "is to say out loud the color of the ink in which the word is printed. " \
                  "The task has two conditions: a congruent words condition, and an " \
                  "incongruent words condition. In the congruent words condition, " \
                  "the words being displayed are color words whose names match the " \
                  "colors in which they are printed: for example RED, BLUE. In the " \
                  "incongruent words condition, the words displayed are color words " \
                  "whose names do not match the colors in which they are printed: for " \
                  "example PURPLE, ORANGE. In each case, we measure the time it takes to " \
                  "name the ink colors in equally-sized lists. Each participant will go " \
                  "through and record a time from each condition."

data = pd.read_csv("./Data/stroopdata.csv", sep=",")
n = float(len(data))
df = 2 * n - 2
print "Sample size: %f" % n
print "Degrees of freedom: %i" % int(n - 1)
D = data['Incongruent'] - data['Congruent']
data['D'] = D
xbar = data.mean()
sbar = data.std()
samples_m = xbar['D']
samples_s = np.sqrt(data.std()['Congruent']**2.0 + data.std()['Incongruent']**2.0)
print samples_m, samples_s

q1 = "Indenpendent variable is the color of the words, i.e. if it conditions are congruent (color and word match) or incongruent (color and word don't match). "
q1 += "The dependent variable time taken to say the color of the ink out loud."

q2 = "The null hypothesis for this data is that there is no difference between the time it takes to say the color of the ink for the congruent and the incongruent. "
q2 += "The alternative hypothesis can either be that the mean time taken to utter the color of the word is different for the two cases "
q2 += "or it could be that time taken for the incongruent case is longer."
q2 += "I am going with the second alternative hypothesis since I think it fits better to the case."
q2 += "I believe that this test measures how fast our brain can differentiate between an answer and the correct answer, when we are reading / seeing the words. "
q2 += "Since the reading of words and the registering of colors is controlled by different areas of the brain, we are essentially "
q2 += "challenging our mind to stick to the task, and therefore need to analyze two inputs before giving an output."
q2 += " Ergo my alternative hypothesis is that the mean time spent on the incongruent case is greater than the mean time spent on the congruent case."
q2 += " Because we don't have the population parameters, I'll be doing a t-test and because I have my set of hypothesis I'll be doing a one-tailed t-test."

q3 = "Since we are analyzing wether the incongruent case increases the time to utter the color, and are using the one-tailed t-test, a descriptive statistic is the "
print data

q4 = "See plots"
fig1 = plt.figure(1, figsize=(14,8))
ax1 = fig1.add_subplot(111)
ax1.hist(data['Congruent'].values, bins=4, normed=1)
ax1.plot(np.linspace(0, 40, 100), mlab.normpdf(np.linspace(0, 40, 100), xbar['Congruent'], sbar['Congruent']))

fig2 = plt.figure(2, figsize=(14,8))
ax2 = fig2.add_subplot(111)
ax2.hist(data['Incongruent'].values, bins=4, normed=1)
ax2.plot(np.linspace(0, 50, 100), mlab.normpdf(np.linspace(0, 50, 100), xbar['Incongruent'], sbar['Incongruent']))

fig3 = plt.figure(3, figsize=(14,8))
ax3 = fig3.add_subplot(111)
ax3.hist(data['D'].values, bins=4, normed=1)
ax3.plot(np.linspace(-30, 30, 100), mlab.normpdf(np.linspace(-30, 30, 100), xbar['D'], sbar['D']))

plt.show()
mydata = [11.154, 21.085]

q5 = ""
t = tstat(xbar['D'], 0, sbar['D'], n)
# alpha = 0.05
# one-tailed
one_tcrit = 1.714
# two-tailed
tcrit = 2.069

print "t-statistic:", t
print "Upper:", samples_m + one_tcrit * sem(samples_s, n)
print "95%% CI:", tconf(samples_m, tcrit, samples_s, n)
#

