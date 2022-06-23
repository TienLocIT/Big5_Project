import { PartialType } from '@nestjs/mapped-types';
import { CreateSuggestQuestionDto } from './create-suggest_question.dto';

export class UpdateSuggestQuestionDto extends PartialType(CreateSuggestQuestionDto) {}
