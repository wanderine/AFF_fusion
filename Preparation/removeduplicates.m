close all
clear all
clc

% This script removes images that are exact duplicates

numDuplicates = 0;

for patient = 1:1124

    files = dir(['/vault/AFF_allfractures/images/patient_' num2str(patient) '_*.png']);

    patient

    % Read all iamges for this patient
    images = {};
    for f = 1:length(files)
        images{f} = imread(['/vault/AFF_allfractures/images/' files(f).name]);
    end

    duplicatematrix = zeros(length(files));

    for m = 1:length(files)

        filenumbers = 1:length(files);
        % Remove current file, to avoid comparison with the same image
        filenumbers = filenumbers(filenumbers ~= m);

        for n = filenumbers

            % Calculate correlation between all pairs of images
            corr = 0;
            try
                corr = corr2(double(images{m}),double(images{n}));
            catch
                disp('Images are of different size')
            end

            if corr == 1
                disp('Found duplicate!')
                duplicatematrix(n,m) = 1;
            end

        end

    end

    % If image 1 and 2 are copies we only want to remove one of them, obtained as upper triangular matrix
    duplicatematrix = triu(duplicatematrix);

    for m = 1:length(files)
        if (sum(duplicatematrix(m,:)) > 0)
            numDuplicates = numDuplicates + 1;
            duplicates{numDuplicates} = files(m).name;
        end
    end

end

length(duplicates)

% Remove duplicate duplicates
duplicates = unique(duplicates);

length(duplicates)


% Remove duplicate files
for f = 1:length(duplicates)
    filename = ['/vault/AFF_allfractures/images/' duplicates{f} ]
    delete(filename)
end





