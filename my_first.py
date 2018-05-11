import survey

def count_delivery_case(objects, predicate):
    return sum( ob.outcome for ob in objects if predicate(ob) )

def get_pregnancy_period(objects, predicate):
    return list(ob.prglength for ob in objects if predicate(ob))

def mean(items):
    return sum(items) / len(items)


if __name__ == '__main__':
    table = survey.Pregnancies()
    table.ReadRecords()
    print('Number of pregnancies ', len(table.records))

    """
    outcome: 정수로 표현된 임신 후 출산 여부. 정상 출산은 1.
    birthord: 정상 출산된 아이의 순서를 정수로 표현한 변수.
        첫 아이의 경우 1, 유산된 경우 0
    """
    print('Normal delivery: ',
        count_delivery_case(table.records, lambda x: x.outcome == 1))
    print('Normal delivery & First-born: ',
        count_delivery_case(
            table.records, lambda x: x.outcome == 1 and x.birthord == 1))
    print('Normal delivery & Other: ',
        count_delivery_case(
            table.records, lambda x: x.outcome == 1 and x.birthord != 1))

    first_child_mean_period = mean(
        get_pregnancy_period(table.records, lambda x: x.birthord == 1))

    other_children_mean_period = mean(
        get_pregnancy_period(table.records,
                            lambda x: x.birthord != 1 and x.outcome == 1))

    diff = (first_child_mean_period - other_children_mean_period) * 7

    print('Mean pregnancy period of first child', first_child_mean_period)
    print('Mmean pregnancy period of other',
          other_children_mean_period)
    print('Diff ', diff)