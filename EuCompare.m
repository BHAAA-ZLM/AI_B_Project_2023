function score = EuCompare(h1, h2)
    assert(length(h1) == length(h2), 'Histogram should have the same length');
    score = sqrt(sum((h1 - h2).^2));
end