function score = IntCompare(h1,h2)
    assert(length(h1) == length(h2), 'Histogram should have the same length');
    score = 1 - sum(min(h1,h2));
end