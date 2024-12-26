function [m, A, eigenfaces, pr_img] = eigenface_core(database_path)
    % List all image files in the database directory
    image_files = dir(fullfile(database_path, '*.jpg'));
    
    % Check if the directory contains any images
    if isempty(image_files)
        error('No images found in the specified database path: %s', database_path);
    end

    % Initialize variables for image processing
    num_images = numel(image_files);
    sample_image = ensure_gray(imread(fullfile(database_path, image_files(1).name)));
    [h, w] = size(sample_image);
    T = zeros(h * w, num_images);

    % Global variables to store names, attendance statuses, and parent emails
    global names;
    global statuses;
    global parent_emails;
    names = cell(num_images, 1);
    statuses = repmat({'Absent'}, num_images, 1); % Initialize all as absent
    parent_emails = cell(num_images, 1); % Initialize parent emails as empty

    % Load images and extract names
    for i = 1:num_images
        image_path = fullfile(database_path, image_files(i).name);
        C = ensure_gray(imread(image_path));
        T(:, i) = C(:);
        
        % Save the name associated with the image
        names{i} = strrep(image_files(i).name, '.jpg', ''); % Remove file extension
        parent_emails{i} = ''; % Initialize parent email as empty
    end

    % Prompt for missing parent emails
    for i = 1:num_images
        if isempty(parent_emails{i})
            prompt_email = sprintf('No parent email found for %s. Please enter the email: ', names{i});
            parent_emails{i} = input(prompt_email, 's');
        end
    end

    % Compute the mean image
    m = mean(T, 2);

    % Subtract the mean from all images
    A = T - m;

    % Compute eigenfaces
    Z = A' * A;
    [V, x] = eig(Z);
    V = V(:, diag(x) > 1); % Select significant eigenvectors
    eigenfaces = A * V;

    % Project images onto the eigenface space
    pr_img = eigenfaces' * A;
end


function face_recognition_and_add(database_path, m, A, eigenfaces, pr_img)
    % Globals for names, statuses, and parent emails
    global names;
    global statuses;
    global parent_emails;

    keep_camera_open = true;

    while keep_camera_open
        % Initialize the webcam
        cam = webcam;
        disp('Camera started. Please look at the camera.');

        % Capture an image from the webcam
        captured_image = snapshot(cam);
        imshow(captured_image);
        title('Captured Image');

        % Process the captured image
        gray_image = ensure_gray(captured_image);
        template_size = [sqrt(size(m, 1)), sqrt(size(m, 1))]; % Ensure consistent template size
        resized_image = resize_to_template(gray_image, template_size);

        % Save the captured image with the recognized or entered name
        X = double(resized_image(:));
        test_image = X - m;
        prtestimg = eigenfaces' * test_image;

        % Compare the captured image with database images
        [p, q] = size(pr_img);
        min_dist = inf;
        closest_index = -1;

        for i = 1:q
            dist = norm(pr_img(:, i) - prtestimg);
            if dist < min_dist
                min_dist = dist;
                closest_index = i;
            end
        end

        % Threshold to determine if the person is new or recognized
        threshold = 8000;
        if min_dist > threshold
            % Add new person to the database
            prompt = 'New person detected. Please enter the name: ';
            new_name = input(prompt, 's');
            prompt_email = 'Please enter the email of a parent or guardian: ';
            parent_email = input(prompt_email, 's');
            new_image_path = fullfile(database_path, [new_name, '.jpg']);
            imwrite(resized_image, new_image_path);
            disp(['New person added: ', new_name, '. Image saved to: ', new_image_path]);

            % Update global variables
            names{q + 1} = new_name;
            statuses{q + 1} = 'Present';
            parent_emails{q + 1} = parent_email;
        else
            % Recognize existing person
            recognized_name = names{closest_index};
            disp(['Person recognized: ', recognized_name]);
            statuses{closest_index} = 'Present';

            % Save the captured image with the recognized name
            recognized_image_path = fullfile(database_path, [recognized_name, '.jpg']);
            imwrite(resized_image, recognized_image_path);
            
            % Prompt for missing parent email if necessary
            if isempty(parent_emails{closest_index})
                prompt_email = sprintf('No parent email found for %s. Please enter the email: ', recognized_name);
                parent_emails{closest_index} = input(prompt_email, 's');
                disp(['Parent email updated for ', recognized_name, '.']);
            end
        end

        % Ask if the camera should stay open
        keep_prompt = 'Do you want to keep the camera open for another person? (yes/no): ';
        response = input(keep_prompt, 's');
        if strcmpi(response, 'no')
            keep_camera_open = false;
        end

        % Release the webcam
        clear cam;
    end

    % Save attendance to a CSV file
    save_attendance_to_csv(names, statuses, parent_emails, fullfile(database_path, 'attendance.csv'));

    % Check for absentees and send emails after a delay
    pause(60);
    send_absent_emails(parent_emails, statuses, names);
end

function send_absent_emails(parent_emails, statuses, names)
    % Identify absentees
    absentees = strcmp(statuses, 'Absent');
    for i = 1:length(names)
        if absentees(i)
            % Prepare email details
            recipient = parent_emails{i};
            if ~isempty(recipient)
                subject = sprintf('Attendance Alert for %s', names{i});
                message = sprintf('Dear Parent/Guardian,\n\nThis is to inform you that %s was marked absent in today''s attendance.\n\nBest regards,\nAttendance System', names{i});
                send_email(recipient, subject, message);
            end
        end
    end
end

function send_email(recipient, subject, message)
    % Configure email preferences
    setpref('Internet','SMTP_Server','smtp.gmail.com');
    setpref('Internet','E_mail','your_email@gmail.com'); % Replace with your email
    setpref('Internet','SMTP_Username','your_email@gmail.com'); % Replace with your email
    setpref('Internet','SMTP_Password','your_password'); % Replace with your password

    % Configure SMTP properties
    props = java.lang.System.getProperties;
    props.setProperty('mail.smtp.auth','true');
    props.setProperty('mail.smtp.starttls.enable','true');
    props.setProperty('mail.smtp.port','587');

    % Send the email
    sendmail(recipient, subject, message);
    disp(['Email sent to: ', recipient]);
end

function gray_image = ensure_gray(image)
    % Convert the image to grayscale if it is RGB
    if size(image, 3) == 3
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end
    % Normalize the grayscale image to the range [0, 1]
    gray_image = mat2gray(gray_image);
end

function resized_image = resize_to_template(image, template_size)
    % Resize the image to match the template size
    resized_image = imresize(image, template_size);
end

function save_attendance_to_csv(names, statuses, parent_emails, filename)
    % Create a table for exporting attendance data
    T = table(names(:), statuses(:), parent_emails(:), 'VariableNames', {'Name', 'Status', 'ParentEmail'});
    writetable(T, filename);
    disp(['Attendance saved to ', filename]);
end

% Main execution block

% Define the database path
database_path = 'D:\MATLAB';

% Perform eigenface training and face recognition
[m, A, eigenfaces, pr_img] = eigenface_core(database_path);

face_recognition_and_add(database_path, m, A, eigenfaces, pr_img);
