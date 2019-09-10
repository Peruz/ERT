ERT data processing
===

A small library for ERT data analysis. Including reading from instrument, filtering, dump all data and analysis to csv, write data for inversion, and plot data information.

Programming Principles
---

Pandas dataframes (DFs) are nice and provide many functionalities.

It is not necessary to change the behavior of DFs, better to add only some functionalities, creating some functions that can be applied to the DF and/or associated numpy arrays (convenient method DF.to_numpy()).
Using composition it is possible to create a ERT class that is composed of 2 DFs, data and electrodes.
Then add the needed methods that apply to these two DFs; note that, in line with composition delegation, it is good to delegate to existing DF methods when possible, increasing code reuse and limiting additional methods that users should spend time to understand.

Make a class when you expect more instances. This helps to keep the data and electrodes associated with the right class.