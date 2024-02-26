# Attitudinally-positioned European sample dataset

## Positioning large online populations across the EU on comparable ideology and issue dimensions

------------------------------------------------------------------------------------------------

Deliverable D2.1 (Work package 2) of the Social Media for Democracy project funded by 
Horizon Europe under Grant No. 101094752.

------------------------------------------------------------------------------------------------

Authors: Pedro Ramaciotti [1,2], Jimena Royo-Letelier [2], 
Jean-Philippe Cointet [2], and Armin Pournaki [2,3]

[1] CNRS Complex Systems Institute of Paris
[2] Sciences Po médialab
[3] Max Planck Institute for Mathematics in Science

------------------------------------------------------------------------------------------------

Cite the dataset as:

Ramaciotti Morales, Pedro; Royo-Letelier, Jimena; Cointet, Jean-Philippe; Pournaki, Armin (2024). Attitudinally-positioned European sample dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.25288210.v1

------------------------------------------------------------------------------------------------

### EXECUTIVE SUMMARY 

These files provide a dataset of attitudinally positioned populations of social media users in Europe. 
Building on recent advancements on
large-scale multi-dimensional political attitude inference in social networks and text, we show how
to create a European sample of attitudinally-positioned users along a Left-Right and a Anti-elite
dimension measuring attitudes towards elites and trust in institutions. These two dimensions are
shown to be relevant to conduct both traditional political analysis on social media and analyses ac-
counting for new forms of polarization related to democratic backsliding. This dataset of users will
serve as a frame of reference for the development of case studies exploring different links between
activity in online platforms, evidence and impacts in politics in other tasks of the project.

------------------------------------------------------------------------------------------------

### BACKGROUND


This document describes the results under Deliverable D2.1 of project SoMe4Dem: “Attitudinally-
positioned European sample dataset”, in which we position large online populations across the EU on
comparable ideology and issue dimensions using social media behavioral trace data. These populations
with multidimensional political positions are then to be used in other tasks in the project.

For the construction of these datasets we begin analyzing Twitter data on the 8 most populous
countries in the EU: Belgium, France, Germany, Italy, the Netherlands, Poland, Romania, Spain, 
and Slovenia

This document describes the protocols and procedure leading to the construction of these datasets,
as well as a description of the data records, its location for open access, and means of validation.

------------------------------------------------------------------------------------------------

DATA RECORDS

These datasets are stored in separate files according to countries. Additionally, for each country, the
positions of MPs and users are stored in separate csv tables. For Belgium, for example, two files are
available: belgium MPs.csv and belgium users.csv. We exclude Romania from the data, because our
labeling strategy in Section 4.4 does not allow us to compute quantitative assessments of the accuracy
of positions on the CHES dimensions.

The files containing information about MPs are structured in the following way:

• Each row is an MPs for which we identified a Twitter account.
• Columns are:
	– delta 1 (specifying the position along the first dimension of the latent homophily space);
	– delta 2 (specifying the position along the second dimension of the latent homophily space);
	– left right (specifying the Left-Right coordinate position of the MP);
	– antielite (specifying the coordinate position of the MP on a scale measuring the salience of
anti-establishment and anti-elite rhetoric);
	– party (specifying the political party to which the MPs is affiliated)

The files containing information about MPs are structured in the following way:
• Each row is a user.
• Columns are:
	– delta 1 (specifying the position along the first dimension of the latent homophily space);
	– delta 2 (specifying the position along the second dimension of the latent homophily space);
	– left right (specifying the Left-Right coordinate position of the user);
	– antielite (specifying the coordinate position of the user on a scale measuring the salience
of anti-establishment and anti-elite rhetoric);
	– labeled left (a binary variable specifying whether the user is labeled as being Left-leaning
by the language model inspecting its text profile bio);
	– labeled right: (a binary variable specifying whether the user is labeled as being Right-leaning
by the language model inspecting its text profile bio);
	– labeled populist: (a binary variable specifying whether the user is labeled as subscribing
populist rhetorics by the language model inspecting its text profile bio);
	– labeled elite: (a binary variable specifying whether the user is labeled as belonging to an
elite group by the language model inspecting its text profile bio).

------------------------------------------------------------------------------------------------

Data AVAILABILITY

Code for analysis of this datasets in available at:

\url{https://doi.org/10.6084/m9.figshare.25288210.v1}


