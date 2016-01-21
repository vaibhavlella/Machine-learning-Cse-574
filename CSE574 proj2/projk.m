%%%%%%%%%%%% LEARNING TO RANK USING LINEAR REGRESSION %%%%%%%%%%%% 

    UBitname = ['k' 'c' 'h' 'a' 'g' 'a' 'n' 't']; 
    personNumber = ['5' '0' '1' '6' '9' '4' '4' '1'];

    clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% REAL WORLD DATA SET %%%%%%%%%%%%
 
% Copy data from the Realtime dataset to .mat file
    fileId                  = fopen('Querylevelnorm.txt');
    str                     = ['%f %*s' repmat('%*d:%f',1,46) '%*[^\n]'];
    matrix                  = textscan(fileId,str,'collectoutput',1);
    matrix                  = cell2mat(matrix);
    filename                = 'project1_data_real.mat';
    save(filename,'matrix');
    fclose(fileId);
    
    fileId=fopen('Querylevelnorm.txt');
    str                      = ['%f %*s' repmat('%*d:%f',1,46) '%*[^\n]'];
    matrix_realdata          = textscan(fileId,str,'collectoutput',1);
    matrix_realdata          = cell2mat(matrix_realdata);
    filename                 ='project1_data_real.mat';
    save(filename,'matrix_realdata');
    fclose(fileId);

    filename                 ='project1_data_real.mat';
    load(filename,'matrix_realdata');
  
    n_train_real             = 55698;
    n_validate_real          = 6962;
    n_test_real              = 6963;
    train_real_input         = matrix_realdata(1:55698,2:47);    % Input train data set
    train_real_target        = matrix_realdata(1:55698,1:1);     % Target train data set
    validation_real_target   = matrix_realdata(55699:62660,1:1);
    validation_real_input    = matrix_realdata(55699:62660,2:47);
    test_real_input          = matrix_realdata(62661:69623,2:47);
    test_real_target         = matrix_realdata(62661:69623,1:1);
    
    
    trainInd(:,:) = 1:55698;
    validInd(:,:) = 55699:62660;
    
    trainInd1=trainInd';
    validInd1=validInd';
    
   

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    M1 = 19;
    %for M1 = 3:14
        
    [idx,mu1] = kmeans(train_real_input,M1);
    [idx2,C2] = kmeans(validation_real_input,M1);
    [idx3,C3] = kmeans(test_real_input,M1);
    
    
    for i = 1:46
        mu1(1,i) = 1;
        C2(1,i)  = 1;
        C3(1,i)  = 1;
    end
    
    var_train_real      = var(train_real_input);
    variance_train_real = var_train_real ;
    
    sigma_train_real    = ((diag(variance_train_real))+(4*eye(46)));
     
    D_real = 46;
    Sigma1 = zeros(D_real,D_real,M1);
    for i= 1:M1
        Sigma1(:,:,i) = sigma_train_real;
    end
    
    phi_train_real = zeros(n_train_real,M1-1);
    phi_train_real = [ones(size(phi_train_real,1),1) phi_train_real];
    
    for j=2:M1
    for z=1:n_train_real
    
        phi_train_real(z,j)=  exp((-1/2)*(train_real_input(z,:)-mu1(j,:))*(sigma_train_real\(train_real_input(z,:)-mu1(j,:))'));
    
    end
    end
    %{
    lambda1 = 0.7;
    w1 = ((lambda1*eye(33,33))+(phi_train_real'*phi_train_real))\((phi_train_real')*(train_real_target));
    hypo_train_real = phi_train_real * w1;
    square_errors_train_real = (train_real_target-hypo_train_real).^2;
    sum_sqr_train_real = sum(square_errors_train_real)/2;
    Eww_train_real = sum(w1.^2)/2;
    Error_reg_train_real = sum_sqr_train_real + lambda1 *(Eww_train_real);
    trainPer1 = sqrt(2*Error_reg_train_real/n_train_real);
    %}
    lambda1 = 3;
    w1 = ((lambda1*eye(M1,M1))+(phi_train_real'*phi_train_real))\((phi_train_real')*(train_real_target));
    hypo_train_real = phi_train_real * w1;
    square_errors_train_real = (train_real_target-hypo_train_real).^2;
    sum_sqr_train_real = sum(square_errors_train_real)/2;
    Error_reg_train_real = sum_sqr_train_real ;
    trainPer1 = sqrt(2*Error_reg_train_real/n_train_real);
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    var_validate_1 = var(validation_real_input);
    variance_validate = var_validate_1 ;
    
    sigma_validate_real = diag(variance_validate)+(4*eye(46));
   
    phi_validate_real = zeros(n_validate_real,M1-1);
    phi_validate_real=[ones(size(phi_validate_real,1),1) phi_validate_real];
    
    for j=2:M1
    for z=1:n_validate_real
    
        phi_validate_real(z,j)=  exp(-(1/2)*(validation_real_input(z,:)-mu1(j,:))*((sigma_train_real)\(validation_real_input(z,:)-mu1(j,:))'));
    
    end
    end
    
    hypo_validate_real = phi_validate_real * w1;
    square_errors_val_real = (validation_real_target-hypo_validate_real).^2;
    sum_sqr_val_real = (sum(square_errors_val_real))/2;
    Error_reg_val_real = sum_sqr_val_real ;
    validPer1 = sqrt(2*Error_reg_val_real/n_validate_real);
    
    %{
    var_test_1 = var(test_real_input);
    variance_test_real = var_test_1 * 0.1;
    
    sigma_test_real = diag(variance_test_real)+(.25*eye(46));
    
    phi_test = zeros(n_test_real,M1-1);
    phi_test=[ones(size(phi_test,1),1) phi_test];
    
    for j=2:M1
    for z=1:n_test_real
     
        phi_test(z,j)=  exp(-(1/2)*(test_real_input(z,:)-C3(j,:))*pinv(sigma_test_real)*(test_real_input(z,:)-C3(j,:))');
    end
    end
     
    Wml_test = ((lambda1*eye(3,3))+(phi_test'*phi_test))\((phi_test')*(test_real_target));
    hypo_test = phi_test * Wml_test;
    square_errors_test = (test_real_target-hypo_test).^2;
    sum_sqr_test = sum(square_errors_test)/2;
    Eww_test = sum(w1.^2)/2;
    Error_reg_test = sum_sqr_test + lambda1*(Eww_test);
    Erms_cfs_test = sqrt(2*Error_reg_test/6963);
    
   
    %}
    
    
   
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% SYNTHETIC DATA SET%%%%%%%%%%%%

    load('synthetic.mat', 't','x');
    x=x';
    D_Syn=10;
    
    train_synthetic_input =  x(1:1400,1:10);
    validate_synthetic_input = x(1401:2000,1:10);
    
    train_synthetic_target =  t(1:1400,:);
    validate_synthetic_target = t(1401:2000,:);
    
    n_train_syn = 1400;
    n_validate_syn = 600;
    
    trainInd_syn(:,:)= 1:1400;
    validInd_syn(:,:)= 1401:2000;
    
    trainInd2 = trainInd_syn';
    validInd2 = validInd_syn';
    
   % validPer2 = zeros();
    
   % for M2 = 3:14
        
    M2=11;
        [idx_syn,mu2]      = kmeans(train_synthetic_input, M2);
        [idx_syn2,mu2_val] = kmeans(validate_synthetic_input, M2);
        
        for i=1:10
        mu2(1,i)=1;
        mu2_val(1,i)=1;
        end
        
        var1_train_syn= var(train_synthetic_input);
        variance_train_syn = var1_train_syn ;
        
        sigma_train_syn= ((diag(variance_train_syn)));
        Sigma2=zeros(D_Syn,D_Syn,M2);
        for i=1:M2
            Sigma2(:,:,i) = sigma_train_syn;
        end
        
        phi_syn_train = zeros(n_train_syn,M2-1);
        phi_syn_train =([ones(size(phi_syn_train,1),1) phi_syn_train]);
        
        for j=2:M2
        for z=1:n_train_syn
          
            phi_syn_train(z,j)=  exp((-1/2)*(train_synthetic_input(z,:)-mu2(j,:))*(sigma_train_syn\(train_synthetic_input(z,:)-mu2(j,:))'));
      
        end
        end
    lambda2 = 3;    
    w2 = ((lambda2*eye(M2,M2))+ (phi_syn_train'*phi_syn_train))\(phi_syn_train' * train_synthetic_target);
    hypo_syn_train = phi_syn_train * w2;
    square_errors_syn_train = (train_synthetic_target-hypo_syn_train).^2;
    sum_sqr_train_syn = sum(square_errors_syn_train)/2;
    Error_reg_train_syn = sum_sqr_train_syn ;
    trainPer2 = sqrt(2*Error_reg_train_syn/n_train_syn);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
       
        var1_validate_syn= var(validate_synthetic_input);
        variance_validate_syn = var1_validate_syn ;
        
        sigma_validate_syn= diag(variance_validate_syn);
                
        phi_syn_validate = zeros(n_validate_syn,M2-1);
        phi_syn_validate =([ones(size(phi_syn_validate,1),1) phi_syn_validate]);
        
        for j=2:M2
        for z=1:n_validate_syn
          
            phi_syn_validate(z,j)=  exp((-1/2)*(validate_synthetic_input(z,:)-mu2(j,:))*(sigma_train_syn\(validate_synthetic_input(z,:)-mu2(j,:))'));
      
        end
        end
        
    %w2_validate = (phi_syn_validate'*phi_syn_validate)\(phi_syn_validate' * validate_synthetic_target);
    hypo_syn_validate = phi_syn_validate * w2;
    square_errors_syn_validate = (validate_synthetic_target-hypo_syn_validate).^2;
    sum_sqr_validate_syn = sum(square_errors_syn_validate)/2;
    Error_reg_validate_syn = sum_sqr_validate_syn ;
    validPer2 = (sqrt(2*Error_reg_validate_syn/n_validate_syn));
   %validPer2(M2) = (sqrt(2*Error_reg_validate_syn/n_validate_syn));
   %end
   % plot(validPer2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    original_mu= mu1;
    mu1=mu1';
   % validInd1 = validInd1';
   % trainInd1 = trainInd1';
    original_mu2= mu2;
    mu2=original_mu2';
    
    save proj2_full.mat;
    clc;clear;
    load('proj2_full.mat','M1', 'Sigma1', 'UBitname', 'lambda1', 'mu1', 'personNumber', 'trainInd1', 'trainPer1', 'validInd1', 'validPer1', 'w1', 'M2', 'Sigma2', 'lambda2', 'mu2', 'trainInd2', 'trainPer2', 'validInd2', 'validPer2', 'w2');
    save proj2.mat;
    
    %{
    subplot(2,2,3)
    plot(validPer1);
    ylabel('validPer1');
    subplot(2,2,4);
    plot(trainPer1);
    ylabel('trainPer1');
    %}
    