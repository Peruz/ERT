import os
import re
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time


try:
    from numba import jit, prange
except ImportError:
    numba_opt = False
else:
    numba_opt = True


class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        '''exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout'''

        assert exists in (True, False, None)
        assert type in ('file', 'dir', 'symlink', None) or hasattr(type, '__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        if string == '-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise argparse.ArgumentError('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise err('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise err('standard input/output (-) not allowed')
        else:
            e = os.path.exists(string)
            if self._exists is True:
                if not e:
                    raise argparse.ArgumentError("path does not exist: '%s'" % string)
                if self._type is None:
                    pass
                elif self._type == 'file':
                    if not os.path.isfile(string):
                        raise argparse.ArgumentError("path is not a file: '%s'" % string)
                elif self._type == 'symlink':
                    if not os.path.symlink(string):
                        raise argparse.ArgumentError("path is not a symlink: '%s'" % string)
                elif self._type == 'dir':
                    if not os.path.isdir(string):
                        raise argparse.ArgumentError("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise argparse.ArgumentError("path not valid: '%s'" % string)
            else:
                if self._exists is False and e:
                    raise argparse.ArgumentError("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or '.'
                if not os.path.isdir(p):
                    raise argparse.ArgumentError("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise argparse.ArgumentError("parent directory does not exist: '%s'" % p)
        return(string)


def get_cmd():
    """ get command line arguments for data processing
    """

    pathtype = PathType(exists=True, type='file')

    parse = argparse.ArgumentParser()

    mains = parse.add_argument_group('mains')
    filters = parse.add_argument_group('filters')
    adjustments = parse.add_argument_group('adjustments')
    outputs = parse.add_argument_group('output')

    # MAIN
    mains.add_argument('-fName', type=str, help='Data file to process')
    mains.add_argument('-fExtension', type=str, help='Data file extension', default='.Data')
    mains.add_argument('-fType', type=str, help='Instrument', default='labrecque')
    mains.add_argument('-plot', dest='plot', action='store_true', help='plot information on data filtering')

    # FILTERS
    # voltage
    filters.add_argument('-v', dest='v', action='store_true', help='perform minimum voltage check')
    filters.add_argument('-v_min', type=float, default=1E-5, help='min voltage value')
    # reciprocal
    filters.add_argument('-rec', dest='rec', action='store_true', help='perform reciprocal check')
    filters.add_argument('-rec_max', type=float, default=10, help='max value for reciprocal error')
    filters.add_argument('-rec_couple', dest='rec_couple', action='store_true', help='couple reciprocal measurements')
    filters.add_argument('-rec_keep_unpaired', dest='rec_keep_unpaired', action='store_true', help='keep measurements without reciprocal')
    # stacking
    filters.add_argument('-stk', dest='stk', action='store_true', help='perform standard deviation check')
    filters.add_argument('-stk_max', type=float, default=10, help='max value from stacking error')

    # rhoa and k
    filters.add_argument('-k', dest='k', action='store_true', help='perform geometric factor check')
    filters.add_argument('-k_max', dest='k_max', type=float, help='maximum geometric factor', default=500)
    filters.add_argument('-k_file', type=pathtype, help='file containing the geometrical factors')  # fromat a b m n r k ...
    filters.add_argument('-rhoa', dest='rhoa', action='store_true', help='perform rhoa check')
    filters.add_argument('-rhoa_min', type=float, default=2, help='min rhoa value')
    filters.add_argument('-rhoa_max', type=float, default=500, help='max rhoa value')
    # contact resistance
    filters.add_argument('-ctc', dest='ctc', action='store_true', help='perform contact resistance check')
    filters.add_argument('-ctc_max', type=float, default=1E+5)  # ohm

    # OUTPUT
    outputs.add_argument('-wrt_rhoa', dest='wrt_rhoa', action='store_true', help='include rhoa column in the output')
    outputs.add_argument('-wrt_ip', dest='wrt_ip', action='store_true', help='include phase column in the output')

    # ADJUSTMENTS
    adjustments.add_argument('-shift_abmn', type=int, default=0, help='shift abmn in data table')
    adjustments.add_argument('-shift_meas', type=int, default=0, help='shift measurement number')
    adjustments.add_argument('-shift_elec', type=int, default=0, help='shift the electrode number in the electrode table')
    # DEFAULTS
    parse.set_defaults(stk=False, rec=False, volt=False, rhoa=False, ctc=False, k=False,
                       rec_couple=False, rec_keep_unpaired=False, wrt_rhoa=False, wrt_ip=False, plot=False)

    args = parse.parse_args()
    return(args)


def check_cmd(args):
    """ make sure cmd arguments agree """

    if args.wrt_rhoa and not args.rhoa:
        raise ValueError('the calculation of rhoa is necessary to add it as a column in the output file, see args.rhoa and args.wrt_rhoa')

    if args.rec_couple and not args.rec:
        raise ValueError('processing the reciprocals is necessary to couple them, see args.rec and args.rec_couple')

    if args.fName == 'all':
        print('all files in current directory will be processed')
    else:
        print(args.fName, 'will be processed')

    return(args)


def find_files(extension):
    """ find all Data files if all the files in the directory have to be processed"""
    print('-' * 80)
    print('Looking for all data files with extension: ', extension)
    all_files = [f for f in os.listdir() if f.endswith(extension)]
    print('Data files found are: ', all_files)
    return(all_files)


def output_files(fname, extension='.dat'):
    """define names for the output file and clean them if already exist"""
    fname_noExtension = os.path.splitext(fname)[0]
    output_fname = fname_noExtension + extension
    try:
        os.remove(output_fname)
    except:
        pass
    return(output_fname)


def read_labrecque(FileName=None):
    """ read a labrecque data file an return data and electrode dataframes"""

    print('-' * 80)
    print('reading ', FileName)

    reex = r'[-+]?[.]?[\d]+[\.]?\d*(?:[eE][-+]?\d+)?'

    # init lists to store the numbers of rows for data, elec, and to skip (invalid data)
    ElecAllColumns, DataAllColumns, lines2skip = [], [], []
    AppRes = False
    # read file to fill lists of data and elec rows, also look for invalid data and data format (AppRes and FreDom)
    with open(FileName) as fid:
        for i, line in enumerate(fid):
            # check electrode rows
            if '#elec_start' in line:
                first_elec = i + 2
            elif '#elec_end' in line:
                last_elec = i
            # check data rows
            elif '#data_start' in line:
                first_data = i + 3
            elif '#data_end' in line:
                last_data = i
            elif 'TX Resist. out of range' in line:
                lines2skip.append(i)
            # check formats
            elif 'Appres' in line:
                print('Apperent Resisitivity column was found')
                AppRes = True
            elif 'FStcks' in line:
                print('Data file in Frequency Domain')
                FreDom = True
            elif 'TStcks' in line:
                print('Data file in Time Domain')
                FreDom = False

        elec_lines = [i for i in range(first_elec, last_elec)]
        data_lines = [i for i in range(first_data, last_data) if i not in lines2skip]

        # go back to the top of the file, to read again, extracting electrodes and data
        fid.seek(0)

        # read electrode and data
        for i, line in enumerate(fid):
            if i in elec_lines:
                nums = re.findall(reex, line)
                ElecAllColumns.append(nums)
            elif i in data_lines:
                nums = re.findall(reex, line)
                DataAllColumns.append(nums)

    # data
    ap = int(AppRes)
    fd = int(FreDom)
    num_meas, a, b, m, n, r, v, stk, ctc, day, time = 0, 2, 4, 6, 8, 9 + ap, 11 + ap + fd * 2, 12 + ap + fd * 2, 15 + ap + fd * 5, -6, -5
    ip = 10 + ap    # define here for structure consistency, then set data[:, ip] = 0
    datanp = np.array(DataAllColumns, dtype=np.float)[:, [num_meas, a, b, m, n, r, ip, v, ctc, stk, day, time]]
    datadf = pd.DataFrame(datanp)
    data_headers = ['meas', 'a', 'b', 'm', 'n', 'r', 'ip', 'v', 'ctc', 'stk', 'day', 'time']
    datadf.rename(columns=dict(zip(range(len(data_headers)), data_headers)), inplace=True)
    datadf['stk'] = abs(datadf['stk'] / datadf['v']) * 100
    # electrodes
    num_elec, x, y, z = 1, 2, 3, 4
    elecnp = np.array(ElecAllColumns, dtype=np.float)[:, [num_elec, x, y, z]]
    elecdf = pd.DataFrame(elecnp)
    elec_headers = ['num', 'x', 'y', 'z']
    elecdf.rename(columns=dict(zip(range(len(elec_headers)), elec_headers)), inplace=True)

    if not FreDom:
        datadf['ip'] = None

    # report invalid lines if present
    if not lines2skip == []:
        print('\n!!! number of invalid data rows (TX Resis. out of range): ', len(lines2skip), '\n    see Data file, continuing with valid data...')

    return(elecdf, datadf)


def read_bert(k_file=None):
    reex = r'[-+]?[.]?[\d]+[\.]?\d*(?:[eE][-+]?\d+)?'

    with open(k_file) as fid:
        lines = fid.readlines()
    elec_num = int(lines[0])
    data_num = int(lines[elec_num + 2])

    elec_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=1, nrows=elec_num, header=None)
    elec = elec_raw[elec_raw.columns[:-1]]
    elec.columns = elec_raw.columns[1:]
    data_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=elec_num + 3, nrows=data_num)
    data = data_raw[data_raw.columns[:-1]]
    data.columns = data_raw.columns[1:]
    return(elec, data)


def fun_rec(a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray, x: np.ndarray):
    l = int(len(x))
    rec_num = np.zeros_like(x)
    rec_avg = np.zeros_like(x)
    rec_err = np.zeros_like(x)
    rec_fnd = np.zeros_like(x)
    for i in range(l):
        if rec_num[i] != 0:
            continue
        for j in range(i + 1, l):
            if (a[i] == m[j] and b[i] == n[j] and m[i] == a[j] and n[i] == b[j]):
                avg = (x[i] + x[j]) / 2
                err = abs(x[i] - x[j]) / abs(avg) * 100
                rec_num[i] = j + 1
                rec_num[j] = i + 1
                rec_avg[i] = avg
                rec_avg[j] = avg
                rec_err[i] = err
                rec_err[j] = err
                rec_fnd[i] = 1  # mark the meas with reciprocals, else leave 0
                rec_fnd[j] = 2  # distinguish between directs and reciprocals
                break
    return(rec_num, rec_avg, rec_err, rec_fnd)

if numba_opt:
    fun_rec = jit(signature_or_function='UniTuple(float64[:],4)(int32[:],int32[:],int32[:],int32[:],float64[:])',
                  nopython=True, parallel=False, cache=True, fastmath=True, nogil=True)(fun_rec)


class ERTdataset():
    """ A dataset class composed of two dataframes data and elec.
    The class does rely on delegation for many functionalities, useful functionalities are:
    * pandas.combine_first : to set data from another dataframe
    * to shift elec nums or meas num, just act on the dataframe specific columns
    * pandas.rename
    * dataframe.to_numpy : to ease and speed calculations (especially row-wise)
    """

    def __init__(self, data=None, elec=None,
                 data_headers=['meas', 'a', 'b', 'm', 'n',
                               'r', 'k', 'rhoa', 'ip', 'v', 'ctc', 'stk', 'day', 'time',
                               'rec_num', 'rec_fnd', 'rec_avg', 'rec_err', 'rec_ip_avg', 'rec_ip_err',
                               'rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid', 'valid'],
                 data_dtypes={'meas': 'Int16', 'a': 'Int16', 'b': 'Int16', 'm': 'Int16', 'n': 'Int16',
                              'r': float, 'k': float, 'rhoa': float, 'ip': float,
                              'v': float, 'ctc': float, 'stk': float, 'day': 'Int64', 'time': 'Int64',
                              'rec_num': 'Int16', 'rec_fnd': bool,
                              'rec_avg': float, 'rec_err': float,
                              'rec_ip_avg': float, 'rec_ip_err': float,
                              'rec_valid': bool, 'k_valid': bool, 'rhoa_valid': bool, 'v_valid': bool,
                              'ctc_valid': bool, 'stk_valid': bool, 'valid': bool},
                 elec_headers=['num', 'x', 'y', 'z'],
                 elec_dtypes={'num': 'Int16', 'x': float, 'y': float, 'z': float}
                 ):

        self.data = None
        self.data_headers = data_headers
        self.data_dtypes = data_dtypes
        self.elec = None
        self.elec_headers = elec_headers
        self.elec_dtypes = elec_dtypes

        if data is not None:
            self.init_EmptyData(data_len=len(data))
            self.data.update(data)
            self.data = self.data.astype(self.data_dtypes)

        if elec is not None:
            self.init_EmptyElec(elec_len=len(elec))
            self.elec.update(elec)
            self.elec = self.elec.astype(self.elec_dtypes)

    def init_EmptyData(self, data_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.data = pd.DataFrame(None, index=range(data_len), columns=range(len(self.data_headers)))
        self.data.rename(columns=dict(zip(range(len(self.data_headers)), self.data_headers)), inplace=True)
        self.data = self.data.astype(self.data_dtypes)

    def init_EmptyElec(self, elec_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.elec = pd.DataFrame(None, index=range(elec_len), columns=range(len(self.elec_headers)))
        self.elec.rename(columns=dict(zip(range(len(self.elec_headers)), self.elec_headers)), inplace=True)
        self.elec = self.elec.astype(self.elec_dtypes)

    def process_rec(self, fun_rec=fun_rec, x='r', x_avg='rec_avg', x_err='rec_err'):
        a = self.data['a'].to_numpy(dtype=int)
        b = self.data['b'].to_numpy(dtype=int)
        m = self.data['m'].to_numpy(dtype=int)
        n = self.data['n'].to_numpy(dtype=int)
        x = self.data[x].to_numpy(dtype=float)
        rec_num, rec_avg, rec_err, rec_fnd = fun_rec(a, b, m, n, x)
        self.data['rec_num'] = rec_num
        self.data['rec_fnd'] = rec_fnd
        self.data[x_avg] = rec_avg
        self.data[x_err] = rec_err

    def get_k(self, data_k):
        if len(self.data) == len(data_k):
            self.data['k'] = data_k['k']
        elif len(data_k) < len(self.data):
            raise IndexError('len k < len data, make sure the right k file is used')
        elif len(self.data) < len(data_k):
            warnings.warn('len k != len data; make sure the right k file is used', category=UserWarning)
            data_k_dtype = {key: self.data_dtypes[key] for key in data_k.columns if key in ['a', 'b', 'm', 'n', 'k']}
            right = data_k[['a', 'b', 'm', 'n', 'k']].astype(data_k_dtype)
            self.data = self.data.merge(data_k[['a', 'b', 'm', 'n', 'k']], on=['a', 'b', 'm', 'n'], how='left', suffixes=('', '_k'), copy=False)
            self.data['k'] = self.data['k_k']
            self.data.drop(columns='k_k', inplace=True)

    def couple_rec(self, couple=False, keep_unpaired=False, dir_mark=1, rec_mark=2, unpaired_mark=0):
        if (not couple and keep_unpaired):  # i.e. keep all, as it is
            return()
        groupby_df = self.data.groupby(self.data['rec_fnd'])
        if (couple and keep_unpaired):
            self.data = pd.concat([groupby_df.get_group(dir_mark), groupby_df.get_group(unpaired_mark)])
        elif(couple and not keep_unpaired):
            self.data = groupby_df.get_group(dir_mark)

    def to_bert(self, fname, wrt_rhoa, wrt_ip, data_columns=['a', 'b', 'm', 'n', 'r'], elec_columns=['x', 'y', 'z']):
        if wrt_rhoa:
            data_columns.append('rhoa')
        if wrt_ip:
            data_columns.append('ip')
        with open(fname, 'a') as file_handle:
            file_handle.write(str(len(self.elec)))
            file_handle.write('\n')
            file_handle.write('# ' + ' '.join(elec_columns) + '\n')
            self.elec[elec_columns].to_csv(file_handle, sep=' ', index=None, line_terminator='\n', header=False)
            file_handle.write(str(len(self.data[self.data.valid == 1])))
            file_handle.write('\n')
            file_handle.write('# ' + ' '.join(data_columns) + '\n')
            self.data[self.data.valid == 1][data_columns].to_csv(file_handle, sep=' ', index=None, line_terminator='\n', header=False)

    def plot(self, fname, plot_columns=['ctc', 'stk', 'v', 'rec_err', 'k', 'rhoa'], valid_column='valid'):
        colors_validity = {1: 'b', 0: 'r'}
        labels_validity = {1: 'Valid', 0: 'Invalid'}
        groupby_df = self.data.groupby(self.data['valid'])
        for key in groupby_df.groups.keys():  # for group 1 (valid) and group 0 (invalid)
            meas = groupby_df.get_group(key)['meas'].to_numpy(dtype=int)
            for c in plot_columns:
                fig_name = fname + labels_validity[key] + '_' + c + '.png'
                plt.plot(meas, groupby_df.get_group(key)[c].to_numpy(), 'o', color=colors_validity[key], markersize=4)
                plt.ylabel(c)
                plt.xlabel('measurement num')
                plt.tight_layout()
                plt.savefig(fig_name)
                plt.close()

    def report(self, report_columns=['ctc_valid', 'stk_valid', 'v_valid', 'rec_valid', 'k_valid', 'rhoa_valid', 'valid']):
        for c in report_columns:
            print('-----\n', self.data[c].value_counts())


def get_options(args_dict: dict):
    """ get options from command-line, update with function args_dict if needed, then check consistency
    """
    args = get_cmd()
    for key, val in args_dict.items():
        if not hasattr(args, key):
            raise AttributeError('unrecognized option: ', key)
        else:
            setattr(args, key, val)
    args = check_cmd(args)
    return(args)


def process(args):
    """ process one or more ERT files """
    # get file(s) to process
    if args.fName == 'all':
        all_files = find_files(args.fExtension)
    else:
        all_files = [args.fName]
    # read file(s)
    for file in all_files:
        args.fName = file
        if args.fType == 'labrecque':
            elec, data = read_labrecque(file)
        # pass to ERTdataset class
        ds = ERTdataset(data=data, elec=elec)
        # adjust
        if args.shift_abmn is not None:
            ds.data[['a', 'b', 'm', 'n']] += args.shift_abmn
        if args.shift_meas is not None:
            ds.data['meas'] += args.shift_meas
        if args.shift_elec is not None:
            ds.elec['num'] += args.shift_elec
        # filters
        if args.rec:
            ds.process_rec()
            if any(ds.data['ip']):
                ds.process_rec(x='ip', x_avg='rec_ip_avg', x_err='rec_ip_err')
            ds.data['rec_valid'] = ds.data['rec_err'] < args.rec_max
        if args.ctc:
            ds.data['ctc_valid'] = ds.data['ctc'] < args.ctc_max
        if args.stk:
            ds.data['stk_valid'] = ds.data['stk'] < args.stk_max
        if args.volt:
            ds.data['v_valid'] = ds.data['v'] > args.v_min
        if (args.k or args.rhoa):  # get k if either k or rhoa are True
            elec_kfile, data_kfile = read_bert(k_file=args.k_file)
            ds.get_k(data_kfile)
        if args.k:
            ds.data['k_valid'] = ds.data['k'].abs() < args.k_max
        if args.rhoa:
            ds.data['rhoa'] = ds.data['r'] * ds.data['k']
            ds.data['rhoa_valid'] = ds.data['rhoa'].between(args.rhoa_min, args.rhoa_max)
        # combine filters
        ds.data['valid'] = ds.data[['rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid']].all(axis='columns')
        output_fname = output_files(args.fName, extension='.csv')
        ds.data.to_csv(output_fname)  # dump all data
        # combine rec for lighter output and inversion
        ds.couple_rec(couple=args.rec_couple, keep_unpaired=args.rec_keep_unpaired)
        # output
        output_fname = output_files(args.fName, extension='.dat')
        ds.to_bert(output_fname, wrt_rhoa=args.wrt_rhoa, wrt_ip=args.wrt_ip)
        ds.report()
        # plot
        if args.plot:
            ds.plot(args.fName)


def main_process(**kargs):
    args = get_options(args_dict=kargs)
    process(args)

if __name__ == '__main__':
    main_process()
