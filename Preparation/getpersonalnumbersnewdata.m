function [personalNumbers, scanDatesAllSubjects, dicomFilesAllSubjects, numberOfDicomFilesAllSubjects] = getpersonalnumbersnewdata(directory)

directories = dir(directory);

personalNumbers = {};
dicomFilesAllSubjects = {};
scanDatesAllSubjects = {};
numberOfDicomFilesAllSubjects = zeros(length(directories)-2,1);

% Loop over directories, each directory corresponds to one subject (one personal number), two first directories are . and .. 
for subject = 3:length(directories)
    
    scanDates = {};
    dicomFiles = {};
    imageCounter = 1;
    
    % Save the personal number as the name of this directory
    personalNumbers{subject-2} = directories(subject).name;

    if length(directories(subject).name) < 12
        disp('Short personal number')
        directories(subject).name
    elseif length(directories(subject).name) > 12
        disp('Long personal number')
        directories(subject).name
    end
    
    % Go to the DICOM directory of this subject
    dicomdirectory = [ directory directories(subject).name '/DICOM/'  ];
    temp1 = dir(dicomdirectory);
    
    % Loop through all sub-directories
    for i = 3:length(temp1)
        
        thisdirectory = [ dicomdirectory '/' temp1(i).name ];
        temp2 = dir(thisdirectory);
        
        % Loop through directories
        for j = 3:length(temp2)
            
            thisdirectory = [ dicomdirectory '/' temp1(i).name '/' temp2(j).name ];
            temp3 = dir(thisdirectory);
            
            for k = 3:length(temp3)
                
                thisdirectory = [ dicomdirectory '/' temp1(i).name '/' temp2(j).name '/' temp3(k).name];
                temp4 = dir(thisdirectory);
                
                for l = 3:length(temp4)
                    
                    thisdirectory = [ dicomdirectory '/' temp1(i).name '/' temp2(j).name '/' temp3(k).name '/' temp4(l).name];
                    temp5 = dir(thisdirectory);
                    
                    % Open all DICOM headers in this directory
                    % NOT Assuming one file per directory here
                    for m = 3:length(temp5)
                      
                        error = 0;
                        try
                            dicomheader = dicominfo([thisdirectory '/'  temp5(m).name]);                            
                        catch me
                            [thisdirectory '/'  temp5(m).name]
                            disp me
                            error = 1;
                        end

                        % Save name of each DICOM file, and the scan date
                        if error == 0
                            dicomFiles{imageCounter} = [thisdirectory '/'  temp5(m).name];
                            scanDates{imageCounter} = dicomheader.StudyDate;
                            imageCounter = imageCounter + 1;
                        end
                                                    
                    end
                    
                end
            end            
        end                
    end
    
    scanDatesAllSubjects{subject-2} = scanDates;
    dicomFilesAllSubjects{subject-2} = dicomFiles;
    numberOfDicomFilesAllSubjects(subject-2) = imageCounter-1;
    
end
