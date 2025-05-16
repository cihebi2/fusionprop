from django import forms
import re
from io import StringIO # Needed for reading uploaded file in memory

class PredictionForm(forms.Form):
    name = forms.CharField(
        max_length=200,
        required=False,
        label="Job Name / Base Name (Optional)",
        help_text="If submitting multiple sequences via text or file, this name can be used as a prefix for results.",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    # Modify sequence field to accept multiple sequences
    sequences_text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'class': 'form-control', 'placeholder': '>Seq1\nSEQUENCE1\n>Seq2\nSEQUENCE2\n...\nOR\nSEQUENCE1\nSEQUENCE2\n...'}),
        label="Protein Sequences (One per line or FASTA format)",
        required=False # Not required if file is provided
    )
    # Add file upload field
    sequence_file = forms.FileField(
        label="Or Upload a File (FASTA or plain text, one sequence per line)",
        required=False, # Not required if text is provided
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )

    def clean(self):
        """
        Validate the form:
        1. Ensure either sequences_text or sequence_file is provided, but not both.
        2. Parse sequences from the provided source.
        3. Validate sequence characters.
        """
        cleaned_data = super().clean()
        sequences_text = cleaned_data.get('sequences_text')
        sequence_file = cleaned_data.get('sequence_file')
        job_name = cleaned_data.get('name', '') # Get optional job name

        if not sequences_text and not sequence_file:
            raise forms.ValidationError("Please provide sequences either in the text box or by uploading a file.", code='required')

        if sequences_text and sequence_file:
            raise forms.ValidationError("Please provide sequences either in the text box or by uploading a file, not both.", code='ambiguous_input')

        sequences = []
        names = []

        try:
            if sequence_file:
                # Process uploaded file
                content = sequence_file.read().decode('utf-8')
                sequences, names = self._parse_sequences(content)
                # Use filename as base job name if job name wasn't provided
                if not job_name:
                    cleaned_data['name'] = sequence_file.name

            elif sequences_text:
                # Process text area input
                sequences, names = self._parse_sequences(sequences_text)

            if not sequences:
                 raise forms.ValidationError("No valid sequences found in the input.", code='no_sequences')

            # Validate all sequences
            for i, seq in enumerate(sequences):
                 # Basic validation (can be enhanced)
                # Commenting out the standard amino acid check as per user request
                # if not re.fullmatch(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq):
                #      error_msg = f"Sequence {i+1} ('{names[i] if names else seq[:10]+'...'}') contains invalid characters. Only standard amino acids are allowed."
                #      raise forms.ValidationError(error_msg, code='invalid_chars')
                if len(seq) < 5: # Example: Add minimum length validation
                     error_msg = f"Sequence {i+1} ('{names[i] if names else seq[:10]+'...'}') is too short (minimum 5 amino acids)."
                     raise forms.ValidationError(error_msg, code='too_short')


            # Store parsed sequences and names in cleaned_data for the view
            cleaned_data['parsed_sequences'] = sequences
            cleaned_data['parsed_names'] = names if names else [f"{job_name}_{i+1}" if job_name else f"Seq_{i+1}" for i in range(len(sequences))]


        except Exception as e:
            # Catch potential parsing errors or other issues
            raise forms.ValidationError(f"Error processing input: {e}", code='processing_error')


        return cleaned_data

    def _parse_sequences(self, content):
        """
        Parses sequences from a string, handling FASTA and plain text formats.
        Returns tuple: (list_of_sequences, list_of_names)
        Names list will be empty if input is not FASTA.
        """
        sequences = []
        names = []
        current_sequence = ""
        current_name = None
        is_fasta = False

        for line in StringIO(content): # Use StringIO to iterate lines easily
            line = line.strip()
            if not line:
                continue # Skip empty lines

            if line.startswith('>'):
                is_fasta = True
                # If we were building a sequence, save it before starting the new one
                if current_sequence:
                    sequences.append(current_sequence.upper())
                    names.append(current_name if current_name else f"Seq_{len(sequences)+1}")

                current_name = line[1:].strip() # Get name after '>'
                current_sequence = ""
            elif is_fasta:
                # Append line to current FASTA sequence (removing potential spaces)
                current_sequence += "".join(line.split())
            else:
                # Assume plain text, one sequence per line (removing potential spaces)
                sequences.append("".join(line.split()).upper())

        # Add the last sequence if it exists (especially for FASTA)
        if current_sequence:
            sequences.append(current_sequence.upper())
            names.append(current_name if current_name else f"Seq_{len(sequences)+1}")

        # If it wasn't FASTA, names list will be empty
        if not is_fasta:
            names = []

        # Basic check for empty sequences after parsing
        sequences = [s for s in sequences if s]

        return sequences, names

# Remove the old clean_sequence method if it exists
# def clean_sequence(self):
#     ...