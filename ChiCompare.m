function score = ChiCompare(h1,h2)
    assert(length(h1) == length(h2), 'Histograms should have the same length');
    epsilon = 1;
    score = sum((h1 - h2).^2 ./ (h1 + h2 + epsilon)) / 2;
end