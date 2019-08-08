import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt


class ert():
    """ process ert data from labrecque instrument
    ===
    it accounts for:
    * max reciprocal err
    * range of apparent resistivity (k file or pybert)
    * max abs geometric factor (k)
    * max contact resistance
    * min voltage difference
    * max stacking err
    * geometric factor
    steps:
    1. parse and check cmd
    1. find and read files
    2. create a table with all data rows
    3. calc and add error to desidered columns
    4. filter on desidered columns
    5. save and write
    """

    def __init__(self):
        self.reex = r'[-+]?[.]?[\d]+[\.]?\d*(?:[eE][-+]?\d+)?'
        self.filter_value = {}
        self.filter_bool = {}

    def _parse_cmd_(self):
        parse = argparse.ArgumentParser()
        parse.add_argument('fData', type = str, help = 'Data file to process')
        parse.add_argument('-keep_all', dest = 'keep_all', action = 'store_true', help = 'keep all data, no processing')
        # pass true if ip is present        
        parse.add_argument('-fdom', dest = 'fdom', action = 'store_true', help = 'labrecque frequency domain format')
        parse.add_argument('-no_fdom', dest = 'fdom', action = 'store_false', help = 'labrecque time domain format')
        # minimum voltage
        parse.add_argument('-volt', dest = 'volt', action = 'store_true', help = 'perform minimum voltage check')
        parse.add_argument('-no_volt', dest = 'volt', action = 'store_false', help = 'skip minimum voltage check')
        parse.add_argument('-volt_min', type = float, default = 1E-5)
        # reciprocity (error check and filter data)
        parse.add_argument('-recip', dest = 'recip', action = 'store_true', help = 'perform reciprocal check')
        parse.add_argument('-no_recip', dest = 'recip', action='store_false', help = 'skip reciprocal check')
        parse.add_argument('-recip_max', type = float, default=7)
        parse.add_argument('-shift_meas', type = int, default=0)
        # reciprocity (average available reciprocal measurements)
        parse.add_argument('-skip_recip', dest = 'skip_recip', action = 'store_true', help = 'keep reciprocal separated')
        parse.add_argument('-keep_unpaired', dest = 'keep_unpaired', action = 'store_true', help = 'keep measurements without reciprocal by setting err = 0; default: False, err = 999')
        # stacking
        parse.add_argument('-stk', dest = 'stk', action = 'store_true', help = 'perform standard deviation check')
        parse.add_argument('-no_stk', dest = 'stk', action = 'store_false', help = 'skip standard deviation check')
        parse.add_argument('-stk_max', type = float, default = 7)
        # rhoa and k
        parse.add_argument('-k_dir', type = str, default = 'geom_factors')
        parse.add_argument('-k_file', type = str, default = 'nobleERT201907.data') # a b m n r, it needs r after n
        # rhoa
        parse.add_argument('-rhoa', dest = 'rhoa', action = 'store_true', help = 'perform rhoa check')
        parse.add_argument('-no_rhoa', dest = 'rhoa', action = 'store_false', help = 'skip rhoa check')
        parse.add_argument('-rhoa_min', type = float, default = 3)
        parse.add_argument('-rhoa_max', type = float, default = 200)
        parse.add_argument('-rhoa_m', type = float, default = 2000)
        # geometric factor k
        parse.add_argument('-k', dest = 'k', action = 'store_true', help = 'perform geometric factor check')
        parse.add_argument('-k_max', dest = 'k_max', type = float, help = 'maximum geometric factor', default = 500)
        # contact resistance
        parse.add_argument('-ctc', dest = 'ctc', action = 'store_true', help = 'perform contact resistance check')
        parse.add_argument('-no_ctc', dest = 'ctc', action = 'store_false', help = 'skip contact resistance check')
        parse.add_argument('-ctc_max', type = float, default = 1E+6) # ohm
        # output
        parse.add_argument('-wrt_rhoa', dest = 'wrt_rhoa', action = 'store_true', help = 'include rhoa column in the output')
        parse.add_argument('-no_wrt_rhoa', dest = 'wrt_rhoa', action = 'store_false', help = 'exclude rhoa column in output')
        # defaults
        parse.set_defaults(keep_all = False, stk = True, recip = True, volt = True, rhoa = False, ctc = True, fdom = False, skip_recip = False, wrt_rhoa = False, k = False, keep_unpaired = False)
        # parse as store the commands as subclass (args) of the main class (ert)
        self.args = parse.parse_args()

    def _check_cmd_(self):
        """make sure cmd arguments agree"""
        print('\n### check_cmd')

        if self.args.keep_all:
            print('\n!!! keep all data option is set True, skipping all the filters...\n')
            self.args.stk = False
            self.args.recip = False
            self.args.skip_recip = True
            self.args.volt = False
            self.args.ctc = False
            self.args.rhoa = False
            self.args.k = False

        if self.args.wrt_rhoa and not self.args.rhoa:
            raise ValueError('the calculation of rhoa is necessary in order to add it as a column in the output file')
        if not self.args.recip:
            print('\n setting skip_recip = True as recip = True is first necessary\n' )
            self.args.skip_recip = True
        if self.args.fData == 'all':
            print('all files in current directory will be processed')
        else:
            print(self.args.fData, 'will be processed')

        
            
    def _find_data_files_(self):
        """find all Data files if all the files in the directory have to be processed"""
        self.all_files = [f for f in os.listdir() if f.endswith('.Data')]
        print('Data files found are: ', self.all_files)

    def _output_files_(self):
        """define names for the output file and clean them if already exist"""
        self.OutputFile = self.args.fData.replace('.Data', 'Procd.dat')
        self.OutputFileDlt = self.args.fData.replace('.Data', 'Dlt.dat')
        # remove previous files if presents
        if os.path.isfile(self.OutputFile):
            os.remove(self.OutputFile)
        if os.path.isfile(self.OutputFileDlt):
            os.remove(self.OutputFileDlt)


    def _read_Data_(self):
        """read data file (self.fData) and return data and elec tables (self.data and self.elec)"""
        ElecAllColumns, DataAllColumns, self.lines2skip = [], [], []
        self.Data_appres_column = False

        # scan the file for sections positions and conformity
        with open (self.args.fData) as fid:
            for i, line in enumerate(fid):
                if '#elec_start' in line:
                    first_elec = i + 2
                elif '#elec_end' in line:
                    last_elec = i 
                elif '#data_start' in line:
                    first_data = i + 3
                elif '#data_end' in line:
                    last_data = i
                elif 'TX Resist. out of range' in line:
                    self.lines2skip.append(i)
                if 'Appres' in line:
                    print('apperent resisitivity column was found')
                    self.Data_appres_column = True

            elec_lines = [i for i in range(first_elec, last_elec)]
            data_lines = [i for i in range(first_data, last_data) if i not in self.lines2skip]
            # go back to the top of the file, to read again, extracting electrodes and data
            fid.seek(0)
            # read electrode and data
            for i, line in enumerate(fid):
                if i in elec_lines:
                    nums = re.findall(self.reex, line)
                    ElecAllColumns.append(nums)
                elif i in data_lines:
                    nums = re.findall(self.reex, line)
                    DataAllColumns.append(nums)
            self.elec = np.array(ElecAllColumns)
            self.data = np.array(DataAllColumns, dtype = float)
        # report invalid lines if present        
        if not self.lines2skip == []:
            print('\n!!! number of invalid data rows (TX Resis. out of range): ', len(self.lines2skip), '\n    see Data file, continuing with valid data...')


    def _init_columns_(self):
        """define and adjust column numbers depending on the read file and cmd args"""
        # data
        if self.Data_appres_column: 
            ap = 1
        else:
            ap = 0
        self.a, self.b, self.m, self.n, self.r, self.v, self.v_stk, self.ctc_r  = 2, 4, 6, 8, 9 + ap, 11 + ap, 12 + ap, 15 + ap
        if self.args.fdom:
            self.p, self.v, self.v_stk, self.ctc_r  = 10 + ap , 13 + ap, 14 + ap, 20 + ap
        # electrodes
        self.x, self.y, self.z = 2, 3, 4

    def _process_recip_(self):
        """find reciprocal and calculate reciprocal error"""
        skip = [] # to skip rows already processed as reciprocal of previous rows
        self.recip_dict = {} # keep memory of the matches
        R_recip_err = np.empty(self.data.shape[0]) # preallocate vector for resistance reciprocal error
        if self.args.fdom:
            P_recip_err = np.empty(self.data.shape[0]) #  preallocate vector for phase reciprocal error
        for i, dr in enumerate(self.data):
            
            if dr[0] in skip:
                continue # skip this row as it was already processed as a reciprocal measurement
            
            try: # find reciprocal
                idx = np.where((self.data[:,self.a] == dr[self.m]) & (self.data[:,self.b] == dr[self.n]) & (self.data[:,self.m] == dr[self.a]) & (self.data[:,self.n] == dr[self.b]))[0][0]
            
            except: # the previous raise an error if there is no reciprocal
                print('!! processing reciprocal errors, but no reciprocal was found for measurement', int(dr[0]))
                print('use -keep_unpaired to choose between keeping and discharging')
                if self.args.keep_unpaired:
                    R_recip_err[i] = 0
                else: 
                    R_recip_err[i] = 999
                continue
            
            rr = self.data[idx, :] # reciprocal row
            
            # calc errors
            R_dif = np.abs(dr[self.r] - rr[self.r])
            R_avg = (dr[self.r] + rr[self.r]) / 2
            recip_err = np.abs(R_dif / R_avg) * 100
            
            # set errors to both measurements
            R_recip_err[i] = recip_err
            R_recip_err[idx] = recip_err
            
            # add reciprocal to skip
            skip.append(rr[0]) 

            # keep that the 2 meas are reciprocal for later
            self.recip_dict[dr[0]] = rr[0]

            if not self.args.fdom:
                continue
            else:
                phase_dif = np.abs(dr[self.p] - rr[self.p])
                phase_avg = (dr[self.p] + rr[self.p]) / 2
                recip_err = np.abs(phase_dif / phase_avg) * 100
                P_recip_err[i] = recip_err
                P_recip_err[idx] = recip_err
        
        self.filter_value['R_recip_err'] = R_recip_err
        self.filter_bool['R_recip_disc'] = self.filter_value['R_recip_err'] > self.args.recip_max
        print('\nreciprocity, number meas. over the limit: ', np.sum(self.filter_bool['R_recip_disc']))

    def _process_volt_(self):
        self.filter_value['volt_err'] = self.args.volt_min / abs(self.data[:, self.v]) #[0, inf]; <1 good, 1= threshold, >1 bad
        self.filter_bool['volt_disc'] = self.filter_value['volt_err'] > 1
        print('\nminimum voltage, number of meas. below the limit: ', np.sum(self.filter_bool['volt_disc']))

    def _process_ctc_(self):
        self.filter_value['ctc_err'] = self.data[:, self.ctc_r] / self.args.ctc_max #[0, inf]; <1 good, 1= threshold, >1 bad
        self.filter_bool['ctc_disc'] = self.filter_value['ctc_err'] > 1
        print('\nmaximum contact resistance, number of meas. above the limit: ', np.sum(self.filter_bool['ctc_disc']))

    def _process_stk_(self):
        self.filter_value['stk_err'] = np.abs(self.data[:, self.v_stk] / self.data[:, self.v]) * 100
        self.filter_bool['stk_disc'] = self.filter_value['stk_err'] > self.args.stk_max
        print('\nmaximum voltage stacking error, number of meas. above the limit: ', np.sum(self.filter_bool['stk_disc']))

    def _process_k_rhoa_(self):
        """
        steps:
        * get self.quadripoles_k (from file or pybert fwd calc)
        * check dimensions
        * calculate rhoa
        * filter rhoa with args limits
        * filter rhoa on statistical distribution
        """
        print('-' * 80)
        print('-' * 80)
        if self.args.rhoa:
            print('# PROCESSING rhoa')
            print('rhoa min: ', self.args.rhoa_min)
            print('rhoa max: ', self.args.rhoa_max)
        if self.args.k:
            print('# PROCESSING geometric factors')
            print('k max: ', self.args.k_max)
        print('\n')
        
        # get quadripoles_k
        print('reading file with geometrical factors: ', self.args.k_file)
        script_dir = os.path.dirname(__file__)
        k_file_dir = os.path.join(script_dir, self.args.k_dir, self.args.k_file)
        with open(k_file_dir) as fid:
            num_elec = int(fid.readline())
        self.quadripoles_k = np.genfromtxt(k_file_dir, skip_header = num_elec + 4, skip_footer = 1)
        self.quadripoles_k[:, 4] = self.quadripoles_k[:, 8] # k column from pybert

        #if self.lines2skip != []:
        #    self.quadripoles_k = np.array([k for i,k in enumerate(self.quadripoles_k) if i not in self.lines2skip])
        # check dimensions        
        print('  number of geometrical factors: ', self.quadripoles_k.shape[0])
        print('  number of data: ', self.data.shape[0])
        self.geom_factors = self.quadripoles_k[:, 4]

        if not self.lines2skip == []:
            print('!!! removing geometrical factors of invalid data lines to match dimensions')
            self.geom_factors = np.delete(self.geom_factors, self.lines2skip, 0)

        if self.geom_factors.shape[0] != self.data.shape[0]:
            raise ValueError('length of geometric factor list does not match the number of data')


        # calculate rhoa
        self.filter_value['rhoa_value'] = self.data[:, self.r] * self.geom_factors
        # filter with args limits
        disc_rhoa_lim = np.any([[self.filter_value['rhoa_value'] < self.args.rhoa_min],  [self.filter_value['rhoa_value'] > self.args.rhoa_max]], axis = 0)[0]
        # filter rhoa on statistical distribution 
        abs_distance = np.abs(self.filter_value['rhoa_value'] - np.median(self.filter_value['rhoa_value']))
        median_abs_distance = np.median(abs_distance)
        s = abs_distance / median_abs_distance if median_abs_distance else 0.
        #disc_rhoa_stats = self.rhoa[s < self.args.rhoa_m]
        disc_rhoa_stats = s > self.args.rhoa_m
        # combine
        if self.args.rhoa:
            self.filter_bool['rhoa_disc'] = np.any([disc_rhoa_lim,  disc_rhoa_stats], axis = 0)
            self.disc_rhoa_bool_all = np.all([disc_rhoa_lim,  disc_rhoa_stats], axis = 0)
            print('\n')
            print('rhoa, out from limits: ', np.sum(disc_rhoa_lim))
            print('rhoa, out from statistical range: ', np.sum(disc_rhoa_stats))
            print('rhoa, common deleted: ', np.sum(self.disc_rhoa_bool_all))
            print('rhoa, total deleted: ', np.sum(self.filter_bool['rhoa_disc']))
            print('-' * 80)
        if self.args.k:
            print('geometric factors')
            self.filter_bool['k_disc'] = np.abs(self.geom_factors) > self.args.k_max
            print('number of geometric factors over the limit (abs values): ', np.sum(self.filter_bool['k_disc']))
            print('-' * 80)

    def _combine_filters_(self):
        """
        combine the different filters in one np bool array (comb_disc)
        """
        comb_disc = np.zeros((self.data.shape[0]))
        for key in self.filter_bool.keys():
            print('add error: ', key)
            comb_disc = np.any([self.filter_bool[key], comb_disc], axis = 0)
        self.filter_bool['comb_disc'] = comb_disc

    def _apply_combined_filter_(self):
        """apply the combined filter to get clean and deleted data"""
        self.all_data = self.data[:, [0, self.a, self.b, self.m, self.n, self.r]]
        print('initial number of measurements: ', self.all_data.shape[0])
        if self.args.rhoa:
            self.all_data = np.hstack((self.all_data, self.filter_value['rhoa_value'][:, None]))
        if self.args.fdom:
            self.all_data = np.hstack((self.all_data, self.data[:, self.p][:, None]))
        ### apply numpy mask comb_disc, get clean and deleted data
        self.clean_data = self.all_data[self.filter_bool['comb_disc'] == 0]
        self.dlt_data = self.all_data[self.filter_bool['comb_disc'] == 1][:, 1:]
        print('number of deleted measurements: ', np.sum(self.filter_bool['comb_disc'], dtype= int))

    def _couple_recip_(self):
        self.average_ncolumns = 5
        if self.args.rhoa:
            self.average_ncolumns += 1
        if self.args.fdom:
            self.average_ncolumns += 1
        
        avg_data = np.zeros((self.clean_data.shape[0], self.average_ncolumns))
        avg_num = 0
            
        for i, r in enumerate(self.clean_data):
            if not r[0] in self.recip_dict:
                continue # no reciprocal was found, leave the avg data row empty (dummy meas)
            
            recip_idx = np.where(self.clean_data[:, 0] == self.recip_dict[r[0]])[0]
            if recip_idx: # both reciprocals are ok, average them
                avg_num += 1
                recip_row = np.squeeze(self.clean_data[self.clean_data[:, 0] == self.recip_dict[r[0]], :])
                recip_r = recip_row[5]
                avg_r = (r[5] + recip_r) / 2

                avg_data[i, [0, 1, 2, 3]] = r[1:5]
                avg_data[i, 4] = avg_r
                
                if self.args.rhoa:
                    recip_rhoa = self.clean_data[self.clean_data[:, 0] == self.recip_dict[r[0]], 6]
                    avg_rhoa = (r[6] + recip_rhoa) / 2
                    avg_data[i, 5] = avg_rhoa

                if self.args.fdom:
                    recip_phase = recip_row[-1]
                    avg_phase = (r[-1] + recip_phase) / 2
                    avg_data[i, -1] = avg_phase

            else: # there was a reciprocal but it has been filtered...
                avg_data[i,:] = r[1:]
        self.avg_data = avg_data[~np.all(avg_data == 0, axis = 1)]
        print('the number of reciprocal measurements that were averaged: ', avg_num)
        print('final number of data rows: ', self.avg_data.shape[0])


    def _prepare_output_data_(self):
        if hasattr(self, 'avg_data'):
            self.output_data = self.avg_data
        elif not hasattr(self, 'data_avg'):
            self.output_data = np.delete(self.clean_data, 0, axis = 1)

    def _save_output_data_(self):
        self.output_data[:, [0,1,2,3]] += self.args.shift_meas
        with open(self.OutputFile, 'ab') as file_handle:
            np.savetxt(file_handle, np.array([self.elec.shape[0]]), fmt = '%i')
            np.savetxt(file_handle, np.array(['# x y z']), fmt = '%s')
            np.savetxt(file_handle, self.elec[:,[2,3,4]], fmt = '%s')
            np.savetxt(file_handle, np.array([self.output_data.shape[0]]), fmt = '%i')
            if not self.args.fdom and not self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n r']), fmt = '%s')
                np.savetxt(file_handle, self.output_data[:, [0,1,2,3,4]], fmt = '%i %i %i %i %f' )
            elif self.args.fdom and self.args.wrt_rhoa: # do not return rhoa, it seems it is not working
                np.savetxt(file_handle, np.array(['# a b m n R rhoa ip']), fmt = '%s')
                np.savetxt(file_handle, self.output_data, fmt = '%i %i %i %i %f %f %f' )
            elif self.args.fdom and not self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n R ip']), fmt = '%s')
                np.savetxt(file_handle, self.output_data, fmt = '%i %i %i %i %f %f' )
            elif not self.args.fdom and self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n r rhoa']), fmt = '%s')
                np.savetxt(file_handle, self.output_data, fmt = '%i %i %i %i %f %f' )
                
        with open(self.OutputFileDlt, 'ab') as file_handle:
            if not self.args.fdom and not self.args.rhoa:
                np.savetxt(file_handle, np.array(['# a b m n r']), fmt = '%s')
                np.savetxt(file_handle, self.dlt_data, fmt = '%i %i %i %i %f' )
            elif self.args.fdom and self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n R rhoa ip']), fmt = '%s')
                np.savetxt(file_handle, self.dlt_data, fmt = '%i %i %i %i %f %f %f' )
            elif self.args.fdom and not self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n R ip']), fmt = '%s')
                np.savetxt(file_handle, self.dlt_data, fmt = '%i %i %i %i %f %f' )
            elif not self.args.fdom and self.args.wrt_rhoa:
                np.savetxt(file_handle, np.array(['# a b m n r rhoa']), fmt = '%s')
                np.savetxt(file_handle, self.dlt_data, fmt = '%i %i %i %i %f %f' )
    
    def _plot_(self, c = 6, filter = 'comb_disc'):
        c = 6 # rhoa
        if self.args.rhoa:
            plt.subplot(2,1,1)
            plt.title('kept (blue) and discharged (red) data, rhoa values')
            plt.plot(self.all_data[self.filter_bool[filter] == 1, 0], self.all_data[self.filter_bool[filter] == 1, c], 'or')
            plt.ylabel('rhoa [Ohm m]')
            plt.subplot(2,1,2)
            plt.plot(self.all_data[self.filter_bool[filter] == 0, 0], self.all_data[self.filter_bool[filter] == 0, c], 'ob')
            plt.ylabel('rhoa [Ohm m]')
            plt.xlabel('meas. num []')
            plt.savefig('data_discharging.png')
            plt.show()
        if self.args.recip:
            plt.plot(self.all_data[:, 0], self.filter_value['R_recip_err'], 'ob')
            plt.plot(self.all_data[self.filter_bool[filter] == 1, 0], self.filter_value['R_recip_err'][self.filter_bool[filter] == 1], 'or')
            plt.plot(self.all_data[self.filter_value['R_recip_err'] == 999, 0], np.ones_like(self.all_data[self.filter_value['R_recip_err'] == 999, 0]) * 999, 'og')
            plt.plot(self.all_data[self.filter_value['R_recip_err'] == 0, 0], np.ones_like(self.all_data[self.filter_value['R_recip_err'] == 0, 0]), 'oy')
            plt.title('reciprocal error [%]')
            plt.yscale('log')
            plt.xlabel('meas. num []')
            plt.ylabel('Resistance reciprocal error [%]')
            plt.savefig('recriprocal_err.png')
            plt.show()
        if self.args.stk:
            plt.plot(self.all_data[:, 0], self.filter_value['stk_err'], 'ob')
            plt.plot(self.all_data[self.filter_bool[filter] == 1, 0], self.filter_value['stk_err'][self.filter_bool[filter] == 1], 'or')
            plt.ylabel('Resistance stacking error [%]')
            plt.xlabel('meas. num []')
            plt.yscale('log')
            plt.title('staking error [%]')
            plt.savefig('stacking_err.png')
            plt.show()
        if self.args.ctc:
            plt.plot(self.all_data[:, 0], self.filter_value['ctc_err'] * self.args.ctc_max, 'ob')
            plt.plot(self.all_data[self.filter_bool[filter] == 1, 0], self.filter_value['ctc_err'][self.filter_bool[filter] == 1] * self.args.ctc_max, 'or')
            plt.ylabel('contact resistance [ohm]')
            plt.xlabel('meas. num []')
            plt.yscale('log')
            plt.title('contact resistance')
            plt.savefig('contact_resistance.png')
            plt.show()


    def _process_single_(self):
        """process file"""
        self._output_files_()
        self._read_Data_()
        self._init_columns_()
        print('-' * 80)
        print('MEASUREMENT ERROR SECTION')
        if self.args.recip:
            self._process_recip_()
        if self.args.volt:
            self._process_volt_()
        if self.args.ctc:
            self._process_ctc_()
        if self.args.stk:
            self._process_stk_()
        if (self.args.rhoa or self.args.k): 
            self._process_k_rhoa_()
        print('-' * 80)
        print('COMBINE ERRORS AND FILTER')
        self._combine_filters_()
        self._apply_combined_filter_()
        if not self.args.skip_recip:
            print('-' * 80)
            print('AVERAGE RECIPROCALS')
            self._couple_recip_()
        self._prepare_output_data_()
        self._save_output_data_()
        if self.args.fData != 'all':
            self._plot_()

    def _process_all_(self):
        """in case fData = all, process all files in directory"""
        self._find_data_files_()
        for f in self.all_files:
            print('='*80)
            self.args.fData = f
            print('\n\nprocessing: ', self.args.fData)
            self._process_single_()

    def process_main(self):
        """run all methods"""
        self._parse_cmd_()
        self._check_cmd_()
        if self.args.fData == 'all':
            print('processing all files in directory')
            self._process_all_()
        else:
            print('processing single file: ', self.args.fData)
            self._process_single_()


if __name__ == "__main__":
    ert = ert()
    ert.process_main()