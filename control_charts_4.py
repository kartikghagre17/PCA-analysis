

#Control Charts for Principle Components 
fig, ax = plt.subplots()
ax.plot(Z1,'-b', marker='o', mec='y',mfc='r' , label="Z1")
ax.plot([3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label="UCL")
ax.plot([-3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label='LCL')
ax.plot([0 for i in range(len(Z1))], "-", color='black',label='CL')
plt.ylabel('$Z_1$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

fig.show()

#classification

logisticRegr = LogisticRegression(solver='lbfgs')
scoring=['accuracy']
scores_lr_full_data = cross_validate(logisticRegr, df, Y,cv=5, scoring=scoring)
scores_lr_Z = cross_validate(logisticRegr, Z, Y,cv=5, scoring=scoring)
scores_lr_Z12 = cross_validate(logisticRegr, Z[:,:2], Y,cv=5, scoring=scoring)

gnb = GaussianNB()
scores_gnb_full_data = cross_validate(gnb, df, Y,cv=5, scoring=scoring)
scores_gnb_Z = cross_validate(gnb, Z, Y,cv=5, scoring=scoring)
scores_gnb_Z12 = cross_validate(gnb, Z[:,:2], Y,cv=5, scoring=scoring)

scores_dict={}
for i in ['fit_time','test_accuracy']:
  scores_dict["lr_full_data " + i ]=scores_lr_full_data[i]
  scores_dict["lr_Z  " + i ]=scores_lr_Z[i]
  scores_dict["lr_Z12 " + i ]=scores_lr_Z12[i]
  scores_dict["gnb_full_data " + i ]=scores_gnb_full_data[i]
  scores_dict["gnb_Z " + i ]=scores_gnb_Z[i]
  scores_dict["gnb_Z12 " + i ]=scores_gnb_Z12[i]

scores_data=pd.DataFrame(scores_dict)
print(scores_data)


#discplay coefficients
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
score = logisticRegr.score(X_test, y_test)
coefficient_full = logisticRegr.coef_

Z_train, Z_test, yz_train, yz_test = train_test_split(Z, Y, test_size=0.2)
logisticRegr_z = LogisticRegression()
logisticRegr_z.fit(Z_train, yz_train)
score_z = logisticRegr_z.score(Z_test, yz_test)
print(score_z)
coefficient_PCA = logisticRegr_z.coef_
np.around(coefficient_full, decimals=2)

np.around(coefficient_PCA, decimals=2)
scores_data.to_excel("T1.xls")