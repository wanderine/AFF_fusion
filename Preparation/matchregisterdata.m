clc
close all
clear all

% This script matches tabular data (register data) with DICOM images through the personal number of each patient (available in the DICOM header)

% Open register data with personal numbers
registerdata = readtable('registerdata.csv');
numSubjects = height(registerdata);

foundImages = zeros(numSubjects,40); % Can have many images per person
foundImagesAndDates = zeros(numSubjects,40); % Can have many images per person
wrongDateImages = zeros(numSubjects,40);
smallImagesAndDates = zeros(numSubjects,40);
personalNumberLengths = zeros(numSubjects,40);

classes = [1];

DXimages = 0;

% Get type of patient, AFF NFF
AFF_status = table2array(registerdata(:,85));

directory1 = '/vault/AFF_allfractures/dicomimages/';

[personalNumbersClass1, scanDatesClass1, DicomFilesClass1, numberOfDicomFilesAllSubjectsClass1] = getpersonalnumbersnewdata(directory1);

personalNumbersAllClasses{1} = personalNumbersClass1;
scanDatesAllClasses{1} = scanDatesClass1;
dicomFilesAllClasses{1} = DicomFilesClass1;

scanNumbers = zeros(numSubjects,1);
imagespersubject = zeros(numSubjects,1);

scanoccasion = 1;
totalImages = 0;
invalidPersonalNumbers = 0;

personalNumberLengthsAllClasses = {};

% Match personal numbers
for subject = 1:numSubjects

    subject

    % Get current personal number in register data
    currentPnr = num2str(registerdata(subject,:).pnr);

    if AFF_status(subject) == 1
        subjectType = 'AFF';
    else
        subjectType = 'CONTROL';
    end

    % Check in all classes
    for class = classes

        % Get all personal numbers in this class
        personalNumbers = personalNumbersAllClasses{class};
        % Get all scan dates in this class
        scanDates = scanDatesAllClasses{class};
        % Get all DICOM files in this class
        dicomFiles = dicomFilesAllClasses{class};

        % Check one personal number at a time
        for s = 1:length(personalNumbers)

            thisPnr = personalNumbers(s);
            thisPnr = thisPnr{1};

            try
                thisPnr = erase(thisPnr,'-'); % Remove any - in personal number
            catch
                invalidPersonalNumbers = invalidPersonalNumbers + 1;
            end

            % Correct subject
            if strcmp(currentPnr,thisPnr)
                
                foundImages(subject,class) = foundImages(subject,class) + 1;

                % Get scan dates for this subject (directory)
                dates = scanDates{s};

                % Get all files for this subject (directory)
                files = dicomFiles{s};                

                imageNumber = 1;
                imageNumberSmall = 1;

                % Loop over dates (files) for this subject / directory
                for file_ = 1:length(files)
                    
                    file = files{file_}; % Get the current file
                    header = dicominfo(file);
                    
                    % Check if there actually is an image...
                    if header.Width > 0

                        foundImagesAndDates(subject,class) = foundImagesAndDates(subject,class) + 1;
                        copyfile(file,['/vault/AFF_allfractures/matchedimages/patient_' num2str(subject) '_' subjectType '_class_' num2str(classes(class)) '_modality_' header.Modality '_scanoccasion_' num2str(scanNumbers(subject)) '_imagenumber_' num2str(imageNumber) '.dcm' ])
                        imageNumber = imageNumber + 1;
                        totalImages = totalImages + 1;
                        imagespersubject(subject) = imagespersubject(subject) + 1;

                        if strcmp(header.Modality,'DX')
                            DXimages = DXimages + 1;
                        end
                    else
                        disp('Too small image')
                        smallImagesAndDates(subject,class) = smallImagesAndDates(subject,class) + 1;                        
                        imageNumberSmall = imageNumberSmall + 1;
                    end
                end

            end
        end
    end
end

