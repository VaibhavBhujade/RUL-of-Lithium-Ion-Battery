# RUL-of-Lithium-Ion-Battery

## Prediction of Remaining Useful Life of Li-ion Battery using Neural Network and Bat-PF Algorithm

### Problem Statement :
Li-ion batteries are widely used in consumer electronics, electric vehicles and space systems. However, a Li-ion battery has a useful life, that means with continuous charge and discharge cycles and material aging, battery performance will continue to decline until it fails to function.
To predict the remaining useful life (RUL) is an effective way to indicate the health of lithium-ion batteries, which will help to improve the reliability and safety of battery-powered systems. Remaining life of a Li-ion battery is also known as battery cycle life, refer to the number of complete charge/discharge cycles that the battery can support before that its capacity falls under 70% of its original capacity.

It is known that capacity of a Li-ion battery is continuously declining after every charge and discharge cycle, and the degradation trend is very consistent. When a battery capacity drops under the failure threshold, the cell is considered to be not usable. Theoretically, it is possible to predict the remaining life of a Li-ion battery by establishing a life model of a battery. A battery life model can have many applications
Remaining useful life (RUL) is the length of time a machine is likely to operate before it requires repair or replacement. By taking RUL into account, engineers can schedule maintenance, optimize operating efficiency, and avoid unplanned downtime. For this reason, estimating RUL is a top priority in predictive maintenance programs

### Scope
Lithium-ion batteries have been broadly used in transportation,aerospace,and defense military applications due to their low self-discharge rate, high operating voltage, long cycle life, and high energy density. The lithium-ion batteries are usually used to provide power for electrical systems, in other words, they store and then release electrical energy through internal electro-chemical reaction.tuut
 However, the battery suffers from side reactions during operation, which leads to materials aging and capacity fade of the battery, and thus cause performance degradation or even catastrophic events of electrical systems.Therefore,predicting the remaining useful life(RUL)of lithium-ion batteries is critical and indispensable for the electrical systems. Accurate RUL prediction can effectively indicate lithium-ion batteries’ health, which could help to provide maintenance plans to ensure the reliability and safety of the systems


### Motivation
Many  catastrophic events have been reported, three examples of serious degradation-related incidents are as follows:
 (1) A Zotye pure electric car made in China spontaneously combusted on 11     April   2011.
 (2) The U.S. National Highway Traffic Safety Administration (NHTSA) subjected a GM Volt to a side-impact crash test on 12 May 2011, during which the batteries suffered a great impact and degraded. Three weeks later, the temperature of the Volt’s lithium-ion battery pack increased sharply, causing spontaneous combustion.
(3) A Tesla Model S made in Norway suddenly caught fire in 2014 when it was charging in the fast charging station.
To avoid such catastrophic incidents caused by the degradation of lithium-ion batteries and to predictively maintain the safety of vehicles, it is of great significance to carry out research on the RUL prognostics of lithium-ion batteries.

### Charge and discharge
Cathode : Lithium Metal Oxide
Anode : Graphite
Electrolyte : Lithium salt of organic solvent
Lithium is a reactive element and is most stable in its metal oxide.
Graphite is loosely bounded so that separated Li+ ions can be stored.
Electrolyte allows only Li+ ions to pass through it. Electrons are not allowed.
For charging, external power is applied.
Lithium metal oxide becomes positively charged electrode and graphite becomes negatively charged electrode.

After applying the external source, electrons move towards graphite through outside circuit while Li+ ions flows through electrolyte towards graphite.
This means cell is charged.
This is the most unstable state of Li+ ions.
If the external power is removed and a load is connected, Li+ moves towards metal oxide as well as the electrons does the same. Thus current generates.

During this process, at cathode there occurs electrolytic oxidation.This degrades the performance of cell.
The reaction between Graphite and Li+ ions surrounded by the electrolyte molecules leads SEI(Solid Electrolyte Interface).
During charging discharging process, the width of this SEI layer goes on increases.
These 2 are one of the main reasons due to which Degradation of battery takes place.
During discharge, lithium ions (Li+) carry the current within the battery from the negative to the positive electrode, through the non-aqueous electrolyte and separator diaphragm.
During charging, an external electrical power source (the charging circuit) applies an over-voltage (a higher voltage than the battery produces, of the same polarity), forcing a charging current to flow within the battery from the positive to the negative electrode, i.e. in the reverse direction of a discharge current under normal conditions. The lithium ions then migrate from the positive to the negative electrode, where they become embedded in the porous electrode material in a process known as intercalation.
Energy losses arising from electrical contact resistance at interfaces between electrode layers and at contacts with current-collectors can be as high as 20% of the entire energy flow of batteries under typical operating conditions.[123]
Procedure
The charging procedures for single Li-ion cells, and complete Li-ion batteries, are slightly different.
A single Li-ion cell is charged in two stages:
Constant current (CC).
Constant voltage (CV).
A Li-ion battery (a set of Li-ion cells in series) is charged in three stages:
Constant current.
Balance (not required once a battery is balanced).
Constant voltage.
During the constant current phase, the charger applies a constant current to the battery at a steadily increasing voltage, until the voltage limit per cell is reached.
During the balance phase, the charger reduces the charging current (or cycles the charging on and off to reduce the average current) while the state of charge of individual cells is brought to the same level by a balancing circuit, until the battery is balanced. Some fast chargers skip this stage. Some chargers accomplish the balance by charging each cell independently.
During the constant voltage phase, the charger applies a voltage equal to the maximum cell voltage times the number of cells in series to the battery, as the current gradually declines towards 0, until the current is below a set threshold of about 3% of initial constant charge current.
Periodic topping charge about once per 500 hours. Top charging is recommended to be initiated when voltage goes below 4.05 V/cell.
Failure to follow current and voltage limitations can result in an explosion
