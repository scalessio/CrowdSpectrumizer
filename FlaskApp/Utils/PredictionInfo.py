
class PredictionInformation:

    def __init__(self):
        self._det_tx_path = ''
        self._snsid = ''
        self._snsname = ''
        self._month = ''
        self._day = ''
        self._nation = ''
        self._technology = ''
        self._startf = ''
        self._endf = ''
        self._freq_start = ''
        self._freq_end = ''
        self._tc_elapsed = []
        self._txdet_elapsed = []

    def initialize(self, json_msg):
        self._snsid = json_msg['snsid']
        self._snsname = json_msg['snsname']
        self._month = json_msg['month']
        self._day = json_msg['day']
        self._nation = json_msg['nation']
        self._technology = json_msg['technology']
        self._startf = int(json_msg['startf'])
        self._endf = int(json_msg['endf'])
        self._freq_start = int(json_msg['freq_start'])
        self._freq_end = int(json_msg['freq_end'])
        self._det_tx_path = './Deployment/FlaskApp/resources/api/' \
                            'detected_transmissions/{snsr_name}/' \
                            '{month}_{day}/FindOpenSSL.cmake/'\
            .format(snsr_name=self.snsname, month=self.month, day=self.day)

    @property
    def det_tx_path(self):
        return self._det_tx_path

    @det_tx_path.setter
    def det_tx_path(self, value):
        self._det_tx_path = value

    @det_tx_path.deleter
    def det_tx_path(self):
        del self._det_tx_path

    @property
    def txdet_elapsed(self):
        return self._txdet_elapsed

    @txdet_elapsed.setter
    def txdet_elapsed(self, value):
        self._txdet_elapsed.append(value)

    @txdet_elapsed.deleter
    def txdet_elapsed(self):
        del self._txdet_elapsed

    @property
    def tc_elapsed(self):
        return self._tc_elapsed

    @tc_elapsed.setter
    def tc_elapsed(self, value):
        self._tc_elapsed.append(value)

    @tc_elapsed.deleter
    def tc_elapsed(self):
        del self._tc_elapsed

    @property
    def snsid(self):
        return self._snsid

    @snsid.setter
    def snsid(self, value):
        self._snsid = value

    @snsid.deleter
    def snsid(self):
        del self._snsid

    @property
    def snsname(self):
        return self._snsname

    @snsname.setter
    def snsname(self, value):
        self._snsname = value

    @snsname.deleter
    def snsname(self):
        del self._snsname

    @property
    def month(self):
        return self._month

    @month.setter
    def month(self, value):
        self._month = value

    @month.deleter
    def month(self):
        del self._month

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = value

    @day.deleter
    def day(self):
        del self._day

    @property
    def nation(self):
        return self._nation

    @nation.setter
    def nation(self, value):
        self.nation = value

    @nation.deleter
    def nation(self):
        del self._nation

    @property
    def technology(self):
        return self._technology

    @technology.setter
    def technology(self, value):
        self._technology = value

    @technology.deleter
    def technology(self):
        del self._technology

    @property
    def startf(self):
        return self._startf

    @startf.setter
    def startf(self, value):
        self._startf = value

    @startf.deleter
    def startf(self):
        del self._startf

    @property
    def endf(self):
        return self._endf

    @endf.setter
    def endf(self, value):
        self._endf = value

    @endf.deleter
    def endf(self):
        del self._endf

    @property
    def freq_end(self):
        return self._freq_end

    @freq_end.setter
    def freq_end(self, value):
        self._freq_end = value

    @freq_end.deleter
    def freq_end(self):
        del self._freq_end

    @property
    def freq_start(self):
        return self._freq_start

    @freq_start.setter
    def freq_start(self, value):
        self._freq_start = value

    @freq_start.deleter
    def freq_start(self):
        del self._freq_start
