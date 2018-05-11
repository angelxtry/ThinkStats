import thinkstats
import my_first
import survey
import math

table = survey.Pregnancies()
table.ReadRecords()
print('Number of pregnancies ', len(table.records))

first_child_pregnancy_periods =\
    my_first.get_pregnancy_period(table.records, lambda x: x.birthord == 1)
first_child_var = thinkstats.Var(first_child_pregnancy_periods)
first_child_stdev = math.sqrt(first_child_var)
print('STDEV of first child: {0}'.format(first_child_stdev))

other_children_pregnancy_periods =\
    my_first.get_pregnancy_period(table.records,
                                  lambda x: x.birthord != 1 and x.outcome == 1)
other_children_var = thinkstats.Var(other_children_pregnancy_periods)
other_children_stdev = math.sqrt(other_children_var)
print('STDEV of other children: {0}'.format(other_children_stdev))