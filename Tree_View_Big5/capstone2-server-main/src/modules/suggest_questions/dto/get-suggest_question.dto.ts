import { IsNotEmpty, IsString } from 'class-validator';
export class GetSuggestQuestionDto {
    @IsString()
    @IsNotEmpty()
    keyword: string;

    @IsString()
    @IsNotEmpty()
    facet: string;
}
