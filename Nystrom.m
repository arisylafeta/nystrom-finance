##############################################################################################
% EXAMPLE USAGE
##############################################################################################

function Nystrom()
    % Main function to setup and run all the methods
    p = 4; % Number of features
    n = 4;  % Number of samples
    rank = 1; % Rank of the data matrix
    X = generate_low_rank_data(p, n, rank); % Generate a low rank data matrix

    % Compute sample covariance matrix
    Sigma_simple = sample_covariance_estimator(X);

    % Nyström covariance estimator
    num_landmarks = 1; % Number of landmark points
    Sigma_nystrom = nystrom_covariance_estimator(X, num_landmarks);

    % Nyström PCA
    k = 1; % Number of principal components
    [Sigma_pca, eigvals_pca, eigvecs_pca] = nystrom_pca(X, num_landmarks);

    % Output Frobenius Norm Difference between the Covariance matrices
    fprintf('Frobenius norm difference between the sample covariance matrix and the Nyström estimator: %.4f%%\n', frobenius_norm_difference(Sigma_simple, Sigma_nystrom));
    fprintf('Frobenius norm difference between the Nystorm Covariance estimator and the Nyström PCA Covariance estimator: %.4f%%\n', frobenius_norm_difference(Sigma_nystrom, Sigma_pca));

    %%%% Calculating the relative error between the eigenvector and eigenvalue %%%%%
    [eigenvectors, eigenvalues_matrix] = eigs(Sigma_nystrom, 1);

    % Extract eigenvalues from the diagonal matrix
    eigenvalues = diag(eigenvalues_matrix);

    % Compare the relative errors between the two
    relative_error_eigenvalues = compare_eigenvalues(eigvals_pca, eigenvalues);
    relative_error_eigenvectors = compare_eigenvectors(eigvecs_pca, eigenvectors);

    fprintf('Relative error of the eigenvalues between Nystrom and PCA Nystrom: %.4f%%\n', relative_error_eigenvalues);
    fprintf('Relative error of the eigenvectors between Nystrom and PCA Nystrom: %.4f%%\n', relative_error_eigenvectors);

    fprintf('Sigma PCA: \n')
    disp(Sigma_pca)
    fprintf('Sigma Nystrom: \n')
    disp(Sigma_nystrom)
endfunction


##############################################################################################
% HELPER FUNCTIONS FOR THE NOTEBOOK
##############################################################################################

% Calculates the Frobenius norm of the difference between two matrices
function result = frobenius_norm_difference(A, B)
    % Calculates the Frobenius norm of the difference between two matrices.
    result = norm(A - B, 'fro') * 100 / norm(A, 'fro') * 100;
endfunction

% Estimates the sample covariance matrix of a data matrix
function covariance_matrix = sample_covariance_estimator(X)
    % Estimates the sample covariance matrix of a data matrix.
    n = size(X, 2);
    covariance_matrix = (X * X') / n;
endfunction

% Generates a low rank data matrix
function A = generate_low_rank_data(n_features, n_samples, rank)
    % Generates a low rank data matrix.
    rank = min([rank, n_features, n_samples]);
    A = randn(n_features, rank) * randn(rank, n_samples);
endfunction

function eigval_relative_error = compare_eigenvalues(eigvals_sample, eigvals_nystrom)
    % Compares the top k eigenvalues from two sets and calculates the relative error.
    eigval_relative_error = norm(eigvals_sample - eigvals_nystrom) / norm(eigvals_sample);
end

function eigvecs_relative_error = compare_eigenvectors(eigvecs_sample, eigvecs_nystrom)
    % Compares the top k eigenvectors from two sets and calculates the Frobenius norm of their difference.
    eigvecs_relative_error = norm(eigvecs_sample - eigvecs_nystrom, 'fro') * 100 / norm(eigvecs_sample, 'fro');
end

##############################################################################################
% NYSTROM COVARIANCE ESTIMATOR
##############################################################################################

% Estimates the covariance matrix of a data matrix using the Nyström method
function Sigma_hat = nystrom_covariance_estimator(X, k)
    % Estimates the covariance matrix of a data matrix using the Nyström method.
    p = size(X, 1); % p: number of features
    n = size(X, 2); % n: number of samples

    % Step 1: Select landmark points (randomly select num_landmarks columns)
    indices = randperm(p, k);
    Y = X(indices, :); % Y.shape = (num_landmarks, n)

    % Step 2: Compute the orthogonal projection matrix P using the pseudoinverse
    YYT = Y * Y';
    YYT_pinv = pinv(YYT);
    P = Y' * (YYT_pinv * Y); % P.shape = (n, n)

    % Step 3: Project data onto the subspace spanned by the landmark points
    X_proj = X * P;

    % Step 4: Construct the Nyström covariance estimator
    Sigma_hat = (X_proj * X') / n; % Sigma_hat.shape = (p, p)
endfunction

##############################################################################################
% NYSTROM PRINCIPAL COMPONENT ANALYSIS
##############################################################################################

% Estimates the principal components of a data matrix using the Nyström method
function [Sigma_hat, eigenvalues, eigenvectors] = nystrom_pca(X, k)
    % Estimates the principal components of a data matrix using the Nyström method.
    [p, n] = size(X);  % p: number of features, n: number of samples

    % Step 1: Select landmark points (randomly select num_landmarks columns)
    indices_I = randperm(p, k);
    Y = X(indices_I, :);  % Y.shape = (num_landmarks, n)

    % Step 2: Define J and X_J
    indices_J = setdiff(1:p, indices_I);
    Z = X(indices_J, :);  % Z.shape = (p - num_landmarks, n)

    % Step 3: Compute Thin SVD of Y
    [U_Y, D_Y, V_Y] = svd(Y, 'econ');
    % Step 4: Construct W_I and W_J
    W_Y = (1 / sqrt(n)) * U_Y * D_Y;
    W_Z = (1 / sqrt(n)) * Z * V_Y;

    % Step 5: Compute W
    W = [W_Y; W_Z]; % Suspecting the error might be here

    % Step 6: Perform thin SVD on W
    [U, Lambda, V] = svd(W, 'econ');

    % Side step: Compute the projection matrix P
    YYT = Y * Y';
    YYT_pinv = pinv(YYT);
    P = Y' * (YYT_pinv * Y);

    % Eigenvalues and eigenvectors
    eigenvalues = Lambda.^2;
    eigenvectors = U;

     % Return only the first k eigenvalues and corresponding eigenvectors
    eigenvalues = eigenvalues(1:k);
    eigenvectors = eigenvectors(:, 1:k);


    % Test the relationships provided in the paper, and document I provided you.
    fprintf('Frobenius Norm Difference for Relation (1): %.4f\n', frobenius_norm_difference(Y * Y', U_Y * D_Y.^2 * U_Y'));
    fprintf('Frobenius Norm Difference for Relation (2): %.4f\n', frobenius_norm_difference(Y * Z', U_Y * D_Y * V_Y' * Z'));
    fprintf('Frobenius Norm Difference for Relation (3): %.4f\n', frobenius_norm_difference(Z * Y', Z * V_Y * D_Y * U_Y'));
    fprintf('Frobenius Norm Difference for Relation (4): %.4f\n', frobenius_norm_difference(P, V_Y * V_Y'));


    % Covariance matrix estimator, should be equal to Nyström covariance estimator
    Sigma_hat = W * W';

    return;
endfunction
