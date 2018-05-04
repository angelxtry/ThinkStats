import survey
table = survey.Pregnancies()
table.ReadRecords()
print('Number of pregnancies ', len(table.records))

def count_delivery_case(objects, predicate):
    return sum( ob.outcome for ob in objects if predicate(ob) )

print(count_delivery_case(table.records, lambda x: x.outcome == 1))
print(count_delivery_case(
    table.records, lambda x: x.outcome == 1 and x.birthord == 1))
print(count_delivery_case(
    table.records, lambda x: x.outcome == 1 and x.birthord != 1))

def get_pregnancy_period(objects, predicate):
    return list(ob.prglength for ob in objects if predicate(ob))

def average(items):
    return sum(items) / len(items)

first_child_avg_period = average(
    get_pregnancy_period(table.records, lambda x: x.birthord == 1))
print('first child average pregnancy period ', first_child_avg_period)

other_children_avg_period = average(
    get_pregnancy_period(table.records,
                         lambda x: x.birthord != 1 and x.outcome == 1))
print('other children average pregnancy period ',
      other_children_avg_period)

diff = (first_child_avg_period - other_children_avg_period) * 7
print('diff ', diff)
