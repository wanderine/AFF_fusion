close all
clear all
clc

% Get all DICOM images
files = dir(['/vault/AFF_Anders/AFF_allfractures/matchedimagesAFF/*.dcm']);

% Read and convert all DICOM images
for f = 1:length(files)

    f   

    file = ['/vault/AFF_Anders/AFF_allfractures/matchedimagesAFF/' files(f).name];

    % Try opening file
    try

        image = dicomread(file);
        image = double(image);
        info = dicominfo(file);

        % Check if image is stored with inverted intensity
        if strcmp(info.PhotometricInterpretation,"MONOCHROME1")
            % Flip intensity
            image = imcomplement(image);
            % Make sure that min is positive
            if min(image(:)) < 0
                image = image + abs(min(image(:)));
            end
            % Rescale intensity
            image = image / max(image(:)) * 50000;
            disp('Flipped intensity')
        else
            % Rescale intensity
            image = image / max(image(:)) * 50000;
        end

        % Change from .dcm to .png
        file(end-3:end) = '.png';

        % Save as 16 bit PNG
        imwrite(uint16(image),['/vault/AFF_Anders/AFF_allfractures/images/'  file ]);

    catch
        disp('Could not open file')
    end

end

