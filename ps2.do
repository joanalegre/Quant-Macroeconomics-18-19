clear
cd "C:\Users\maria\Desktop\MÁSTER\Segundo año\PRIMER CUATRIMESTRE\Quantitative Macroeconomics\Problem sets\ps2"
use data.dta

**************************Exercise 1******************************

* We create the labor share in three different ways we saw in class:

gen naive_1 = CE/Y
gen ls_1 = ((PI)*(1-(RI+CP+NI+T-S)/(Y-PI))+CE)/Y
gen ls_2 = (CE + PI)/Y

line naive_1 ls_1 ls_2 year, saving(ex1)

**************************Exercise 2******************************

/* Since for corporate sector there are not included proprietors data, we 
only can calculate the labor share with the naive approximation */

gen naive_2 = ce/cb

* Naive share 2 (it must be the same as the first one):

gen naive_3 = 1 - (cp + r + t)/cb

line naive_2 naive_3 year, saving(ex2)

* What we can see is that both labor shares are practically the same


* Now we want to compare the naive shares for first and second exercises:

line naive_1 naive_2 year, saving(ex3)

**************************Exercise 3.1 - Spain******************************

gen SE = (self/(ocu-self))
gen rat = self/ocu
gen naive_4 = WC/PIB
gen ls_3 = (WC+(SE*WC))/PIB
gen ls_4 = (WC+(rat*PIB))/PIB

line naive_4 ls_3 ls_4 realone Year, saving(ex4)

**********************Exercise 3.2 Corporate - Spain**************************

gen LsCorp = labcorp/ycorp

line LsCorp Year, saving(ex5) 

gr combine ex1.gph ex2.gph ex3.gph ex4.gph ex5.gph, title("Labor shares for US and Spain")
