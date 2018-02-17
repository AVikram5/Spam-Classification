function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

%Initialization of feature vector with all zeroes
x = zeros(n, 1);


%binary feature vector that indicates whether a particular word occurs in the email. That is, x(i) = 1 when word i is present in the email. 


for i = 1 : length(word_indices)
    x(word_indices(i)) = 1;
end
    

end
