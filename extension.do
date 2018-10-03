clear
cd "C:\Users\Joana\Desktop\Cole\18-19\Quant\ps2\definitivo\extention"
use dataextension.dta

**************************Extention with gross labor share for U.S.A******************************

*unambigous capital incom (UCI)
gen UCI = Rincom+Cprof+Ninteres+CsurplusGov 

*Unambigous income (UI)
gen UI = UCI+DEP+CE

*Percentatge of capital that we will use to estimate the capital part of ambiguos income(thita).
gen thita = (UCI+DEP)/UI

*Ambigous income (AI), Computed without statistical discrepancy.
gen AI = PI+T-S+Bctrans

*Ambigous capital income (ACI)
gen ACI = AI*thita

*Capital income (CI)
gen CI = ACI+DEP+UCI

*Output (Y).
gen Y = UCI+DEP+CE+AI

*Gross labor share (LSgross)
gen LSgross = 1-(CI/Y)

*Plot LSgross:
line LSgross Year, saving(grossLS) 


